from typing import List, cast
import pandas as pd
from utils.helpers import load_files
from utils.logger import logger
from learntools.time_series.utils import (
    make_lags,
    make_leads,
)

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

combined_sales = combined_sales.join(oil)
combined_sales["dcoilwtico"] = (
    combined_sales["dcoilwtico"].bfill().ffill().astype("float32")
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

combined_sales = (
    combined_sales.join(lead_oil)
    .join(lead_holiday)
    .join(lead_promotion)
    .join(lag_target)
    .join(lag_promotion)
    .join(lag_holiday)
)

combined_sales_dummified = pd.get_dummies(combined_sales, drop_first=True)

# TODO: Implement below

# --- STANDARDIZE DATA ---
logger.warning("Standardizing data...")

# --- CHECK FEATURES ---
logger.warning("Checking features...")

# --- SPLIT DATA ---
logger.warning("Splitting data...")

# --- TRAIN MODEL ---
logger.warning("Training model...")

# --- EVALUATE MODEL ---
logger.warning("Evaluating model...")

# --- PREDICT FUTURE ---
logger.warning("Predicting future...")
