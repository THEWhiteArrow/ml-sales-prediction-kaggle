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

# --- LOAD DATA ---
logger.info("Loading data...")
holidays_events, oil, stores, transactions, store_sales, query = load_files()

store_sales = store_sales.set_index(["date", "store_nbr", "family"])
query = query.set_index(["date", "store_nbr", "family"])

# --- CLEAN DATA ---
logger.info("Cleaning data...")
unique_families = store_sales.index.get_level_values("family").unique()
unique_stores = store_sales.index.get_level_values("store_nbr").unique()
combined_sales = pd.concat([store_sales, query], axis=0, join="outer").unstack(
    ["store_nbr", "family"]  # type: ignore
)
for store in unique_stores:
    for family in unique_families:
        combined_sales["sales", store, family] = combined_sales[
            "sales", store, family
        ].ffill()
combined_sales = combined_sales.stack(["store_nbr", "family"], future_stack=True)  # type: ignore
combined_sales["id"] = combined_sales["id"].fillna(0.0).astype("uint32")
combined_sales = cast(pd.DataFrame, combined_sales)

# --- FEATURE ENGINEERING ---
logger.info("Feature engineering...")
combined_sales["month"] = combined_sales.index.get_level_values("date").month  # type: ignore
combined_sales["day_of_week"] = combined_sales.index.get_level_values("date").dayofweek  # type: ignore
combined_sales["year"] = combined_sales.index.get_level_values("date").year  # type: ignore


def get_days_since_last_paycheck(period: pd.Period) -> int:
    if period.day == 15 or period.day == period.days_in_month:
        return 1
    elif period.day < 15:
        return period.day + 1
    else:
        return period.day - 15 + 1


def prepare_days_since_last_paycheck(date: pd.PeriodIndex) -> List[int]:
    return [get_days_since_last_paycheck(period) for period in date]


# --- NOTICE ---
# will allow for linear reference but not for non-linear reference
combined_sales["days_since_last_paycheck"] = prepare_days_since_last_paycheck(
    combined_sales.index.get_level_values("date")  # type: ignore
)
combined_sales["days_since_last_paycheck"] = combined_sales[
    "days_since_last_paycheck"
].astype("uint8")

combined_sales["is_holiday"] = combined_sales.index.get_level_values("date").isin(
    holidays_events[
        holidays_events["transferred"].eq(False)
        & holidays_events["locale"].isin(["National"])
    ].index
)

combined_sales["is_holiday"] = combined_sales["is_holiday"].astype("category")
combined_sales["month"] = combined_sales["month"].astype("category")
combined_sales["day_of_week"] = combined_sales["day_of_week"].astype("category")
combined_sales["year"] = combined_sales["year"].astype("category")

combined_sales = combined_sales.join(oil)
combined_sales["dcoilwtico"] = (
    combined_sales["dcoilwtico"].bfill().ffill().astype("float32")
)

# --- NOTICE ---
# A magnitude 7.8 earthquake struck Ecuador on April 16, 2016.
# People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.
# --- NOTICE ---
# pd.perdiod_range must be used to create a range of dates because the dates are treated as period
EARTH_QUAKE_IMPACT_PERIOD = 90
combined_sales["eartquake_impact"] = combined_sales.index.get_level_values("date").isin(
    pd.period_range("2016-04-16", periods=EARTH_QUAKE_IMPACT_PERIOD)
)
lead_oil = make_leads(combined_sales["dcoilwtico"], leads=16, name="dcoilwtico").ffill()
lead_holiday = make_leads(
    combined_sales["is_holiday"], leads=16, name="is_holiday"
).ffill()
lead_promotion = make_leads(
    combined_sales["onpromotion"], leads=16, name="onpromotion"
).ffill()
lag_target = make_lags(combined_sales["sales"], lags=4, name="sales").bfill()
lag_promotion = make_lags(
    combined_sales["onpromotion"], lags=2, name="onpromotion"
).bfill()
lag_holiday = make_lags(combined_sales["is_holiday"], lags=2, name="is_holiday").bfill()

combined_sales_lead_lag = (
    combined_sales.join(lead_oil)
    .join(lead_holiday)
    .join(lead_promotion)
    .join(lag_target)
    .join(lag_promotion)
    .join(lag_holiday)
)


# --- STANDARDIZE AND ENCODE DATA ---
logger.info("Standardizing and encoding data...")
col_to_scale = [
    "dcoilwtico",
    "sales",
    "onpromotion",
    "days_since_last_paycheck",
    *lead_oil.columns,
    *lag_target.columns,
    *lag_promotion.columns,
    *lead_promotion.columns,
]
col_to_encode = [
    "month",
    "day_of_week",
    "year",
    "is_holiday",
    "eartquake_impact",
    *lead_holiday.columns,
    *lag_holiday.columns,
]

scaler = StandardScaler()
scaler.fit(combined_sales_lead_lag[col_to_scale])
col_scaled = pd.DataFrame(
    scaler.transform(combined_sales_lead_lag[col_to_scale]),  # type: ignore
    columns=col_to_scale,
    index=combined_sales_lead_lag[col_to_scale].index,
)
col_encoded = pd.get_dummies(combined_sales_lead_lag[col_to_encode], drop_first=True)  # type: ignore

combined_sales_final = pd.concat(
    [col_scaled, col_encoded, combined_sales_lead_lag["id"]], axis=1
)


# --- CHECK FEATURES ---
logger.info("Checking features...")

DISABLE_CHECKING = True
omit_col = [*combined_sales.select_dtypes("category").columns, "id"]
num_col = list(set(list(combined_sales.columns)) - set(omit_col))

if not DISABLE_CHECKING:

    fig1 = plt.figure(figsize=(6, 4))
    fig1.suptitle("Feature Correlation")

    sns.heatmap(
        combined_sales[num_col].corr(method="spearman"), vmin=-1, vmax=1, center=0, annot=True  # type: ignore
    )

    fig2, ax = plt.subplots(1, len(num_col), figsize=(10, 4))
    fig2.suptitle("Feature Distribution")
    ax = cast(Axes, ax)
    for i in range(len(num_col)):
        ax[i].hist(combined_sales[num_col[i]])  # type: ignore
        ax[i].set_xlabel(num_col[i])  # type: ignore

    plt.show()
else:
    logger.warning("Checking features disabled. Skipping...")
    logger.warning("To enable, set DISABLE_CHECKING to False")


# --- SPLIT DATA ---
logger.info("Splitting data...")

TRAIN_START = "2013-01-01"
TRAIN_END = "2016-12-31"
TEST_START = "2017-01-01"
TEST_END = "2017-07-31"

X = combined_sales_final.drop(columns=["sales", "id"])
y = make_multistep_target(combined_sales_final["sales"], steps=16)

X_train, y_train = X.loc[TRAIN_START:TRAIN_END], y.loc[TRAIN_START:TRAIN_END]
X_test, y_test = X.loc[TEST_START:TEST_END], y.loc[TEST_START:TEST_END]

# --- TRAIN MODEL WITH OPTUNA ---
logger.error("Training model...")


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 0.01, 10.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 100.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 100.0),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-8, 100.0),
    }

    model = RegressorChain(XGBRegressor(**params))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # type: ignore

best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# final_model = RegressorChain(XGBRegressor(**best_params))
# final_model.fit(X_train, y_train)
# --- PREDICT FUTURE ---
logger.error("Predicting future...")
