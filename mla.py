from typing import List, cast
from matplotlib.axes import Axes
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
import optuna
import pandas as pd
from matplotlib import pyplot as plt
from learntools.time_series.utils import make_lags, make_leads, make_multistep_target
import seaborn as sns
from utils.helpers import load_files
from utils.logger import logger

logger.name = __name__

# --- DATA PREPARATION ---
logger.info("Preparing data...")
holidays_events, oil, stores, transactions, store_sales, query = load_files()
main_index = ["store_nbr", "family", "date"]
secondary_index = ["store_nbr", "family"]
unique_families = store_sales["family"].unique()
unique_stores = store_sales["store_nbr"].unique()
data = pd.concat([store_sales, query], axis=0)
data = data.set_index("date")
data = data.join(oil, on="date").rename(columns={"dcoilwtico": "oil"})
holidays_to_consider = (
    holidays_events[
        (holidays_events["transferred"].eq(False))
        & holidays_events["locale"].isin(["National"])
    ]
    .reset_index()
    .drop_duplicates(keep="first", subset=["date"])
    .set_index("date")
)
data["is_holiday"] = data.index.isin(holidays_to_consider.index)


# --- DATA CLEANING ---
logger.info("Cleaning data...")
# --- NOTICE ---
# missing values in sales, oil, and id columns -> oil column needs to be present
data["oil"] = data["oil"].ffill().bfill()
data = data.reset_index().set_index(main_index)


# --- FEATURE ENGINEERING ---
logger.info("Feature engineering...")
# --- NOTICE ---
# 8-step lag target, 16-step lead oil, 16-step lead onpromotion
# day_of_week, month, year
# days_since_last_paycheck, earthquake_impact
# NOT USED: 16-step lead is_holiday
data["day_of_week"] = data.index.get_level_values("date").dayofweek  # type: ignore
data["month"] = data.index.get_level_values("date").month  # type: ignore
data["year"] = data.index.get_level_values("date").year  # type: ignore


def get_days_since_last_paycheck(period: pd.Period) -> int:
    if period.day == 15 or period.day == period.days_in_month:
        return 1
    elif period.day < 15:
        return period.day + 1
    else:
        return period.day - 15 + 1


def prepare_days_since_last_paycheck(date: pd.PeriodIndex) -> List[int]:
    return [get_days_since_last_paycheck(period) for period in date]


data["days_since_last_paycheck"] = prepare_days_since_last_paycheck(date=data.index.get_level_values("date"))  # type: ignore

data["earthquake_impact"] = data.index.get_level_values("date").isin(
    pd.period_range("2016-04-16", periods=90, freq="D")
)

# --- NOTICE ---
# it is very easy to make a mistake here, so be careful -> depending of how the data is structured you need to do proper lgging and leading
# lagging and leading was issue since it was doing it wrongly while in groups


# lag_sales = make_lags(data["sales"], lags=8, name="sales")
# lead_oil = make_leads(data["oil"], leads=16, name="oil")
# lead_onpromotion = make_leads(data["onpromotion"], leads=16, name="onpromotion")
# data_lead_lag = data.join(lag_sales).join(lead_oil).join(lead_onpromotion)

lags = [1, 2, 3, 4, 5]


def make_lags_in_grouped_df(
    df: pd.DataFrame,
    groupby: List[str] = [],
    lag_value: str = "",
    lags: int = -1,
    name: str = "y",
    lags_list: List[int] = [],
) -> pd.DataFrame:

    if lags > 0:
        lags_list.append(lags)

    def create_lagged_columns(group):
        for lag in lags_list:
            group[f"{name}_lag_{lag}"] = group[lag_value].shift(lag)
        return group

    lagged_df = cast(pd.DataFrame, df.apply(create_lagged_columns))

    return lagged_df


data_sales_lagged = make_lags_in_grouped_df(
    data, groupby=secondary_index, lag_value="sales", lags_list=lags
)

# print(data_sales_lagged.loc["1", "AUTOMOTIVE", "2017"][["sales", "sales_lag_1", "sales_lag_2"]])  # type: ignore

# --- STANDARDIZE AND ENCODE ---
logger.info("Standardizing and encoding...")
# --- MODEL TRAINING ---

logger.info("Training model...")
