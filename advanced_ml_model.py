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

submission = pd.read_csv("advanced_ml_model_submission.csv")
submission["sales"] = submission["sales"].clip(0.0)
submission.to_csv("advanced_ml_model_submission_2.csv", index=False)

# --- CONFIG ---
DISABLE_CHECKING = True
CONSIDER_HOLIDAYS = False
CONSIDER_ONPROMOTION = False
EARTH_QUAKE_IMPACT_PERIOD = 90
OPTIMIZE_WITH_OPTUNA = False

TRAIN_START = "2015-01-01"
TRAIN_END = "2017-06-30"
TEST_START = "2017-07-01"
TEST_END = "2017-07-31"
QUERY_START = "2017-08-01"
QUERY_END = "2017-08-15"

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


if CONSIDER_HOLIDAYS:
    combined_sales["is_holiday"] = combined_sales.index.get_level_values("date").isin(
        holidays_events[
            holidays_events["transferred"].eq(False)
            & holidays_events["locale"].isin(["National"])
        ].index
    )

    combined_sales["is_holiday"] = combined_sales["is_holiday"].astype("category")
    lag_holiday = make_lags(
        combined_sales["is_holiday"], lags=2, name="is_holiday"
    ).bfill()
    lead_holiday = make_leads(
        combined_sales["is_holiday"], leads=16, name="is_holiday"
    ).ffill()


if CONSIDER_ONPROMOTION:
    lag_promotion = make_lags(
        combined_sales["onpromotion"], lags=2, name="onpromotion"
    ).bfill()
    lead_promotion = make_leads(
        combined_sales["onpromotion"], leads=16, name="onpromotion"
    ).ffill()
else:
    combined_sales = combined_sales.drop(columns=["onpromotion"])


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

combined_sales["eartquake_impact"] = combined_sales.index.get_level_values("date").isin(
    pd.period_range("2016-04-16", periods=EARTH_QUAKE_IMPACT_PERIOD)
)
lead_oil = make_leads(combined_sales["dcoilwtico"], leads=16, name="dcoilwtico").ffill()


lag_target = make_lags(combined_sales["sales"], lags=4, name="sales").bfill()


combined_sales_lead_lag = combined_sales.join(lead_oil).join(lag_target)

if CONSIDER_HOLIDAYS:
    combined_sales_lead_lag = combined_sales_lead_lag.join(lead_holiday).join(
        lag_holiday
    )

if CONSIDER_ONPROMOTION:
    combined_sales_lead_lag = combined_sales_lead_lag.join(lead_promotion).join(
        lag_promotion
    )


# --- STANDARDIZE AND ENCODE DATA ---
logger.info("Standardizing and encoding data...")

col_to_scale = [
    "dcoilwtico",
    "days_since_last_paycheck",
    *lead_oil.columns,
    *lag_target.columns,
]

col_to_passthrough = [
    "id",
    "sales",
]


col_to_encode = [
    "month",
    "day_of_week",
    "year",
    "eartquake_impact",
]

if CONSIDER_ONPROMOTION:
    col_to_encode = [
        *col_to_encode,
        "onpromotion",
        *lead_promotion.columns,
        *lag_promotion.columns,
    ]

if CONSIDER_HOLIDAYS:
    col_to_encode = [
        *col_to_encode,
        "is_holiday",
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
    [col_scaled, col_encoded, combined_sales_lead_lag[col_to_passthrough]], axis=1
)


# --- CHECK FEATURES ---
logger.info("Checking features...")

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


X = combined_sales_final.drop(columns=["sales", "id"])
y = make_multistep_target(combined_sales_final["sales"], steps=16)

X_train, y_train = X.loc[TRAIN_START:TRAIN_END], y.loc[TRAIN_START:TRAIN_END]
X_test, y_test = X.loc[TEST_START:TEST_END], y.loc[TEST_START:TEST_END]

X_final, y_final = X.loc[TRAIN_START:TEST_END], y.loc[TRAIN_START:TEST_END]
X_query = X.loc[QUERY_START:QUERY_END]

# --- TRAIN MODEL WITH OPTUNA ---
logger.info("Training model...")

if OPTIMIZE_WITH_OPTUNA:

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, log=True),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.01, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-8, 100.0, log=True
            ),
        }

        model = RegressorChain(XGBRegressor(**params))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        return rmse

    study = optuna.create_study(direction="minimize", study_name="RegressorChain")
    study.optimize(objective, n_trials=5)  # type: ignore

    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

# --- NOTICE ---
# Best hyperparameters (0.68): {'n_estimators': 700, 'max_depth': 9, 'learning_rate': 0.041849257885856056, 'subsample': 0.8288398810318013, 'colsample_bytree': 0.8544896785347699, 'gamma': 1.5166768979741576, 'reg_alpha': 0.9719392708524538, 'reg_lambda': 32.62483746686035, 'min_child_weight': 2.6631715214066274}
legacy_best_params = {
    "n_estimators": 700,
    "max_depth": 9,
    "learning_rate": 0.041849257885856056,
    "subsample": 0.8288398810318013,
    "colsample_bytree": 0.8544896785347699,
    "gamma": 1.5166768979741576,
    "reg_alpha": 0.9719392708524538,
    "reg_lambda": 32.62483746686035,
    "min_child_weight": 2.6631715214066274,
}
final_model = RegressorChain(XGBRegressor(**legacy_best_params))
final_model.fit(X_final, y_final)

# --- PREDICT FUTURE ---
logger.info("Predicting future...")
y_query = pd.DataFrame(
    final_model.predict(X_query), index=X_query.index, columns=y_final.columns  # type: ignore
)

# --- SAVE PREDICTION ---
query_families = query.index.get_level_values("family").unique()
query_stores = query.index.get_level_values("store_nbr").unique()
y_query_converted = pd.DataFrame()

for i in range(16):
    y_query_converted = pd.concat(
        [
            y_query_converted,
            pd.concat(
                [
                    y_query.loc["2017-08-15"]
                    .reset_index()[["store_nbr", "family", f"y_step_{i+1}"]]
                    .rename(columns={f"y_step_{i+1}": "sales"}),
                    pd.Series(
                        [f"2017-08-{16+i}"] * len(query_families) * len(query_stores),
                        name="date",
                        dtype="period[D]",
                    ),
                ],
                axis=1,
            ),
        ],
        axis=0,
    )


logger.error("Saving prediction...")
y_submission = (
    query["id"]
    .to_frame()
    .join(y_query_converted.set_index(["date", "store_nbr", "family"]))
)
y_submission.to_csv("advanced_ml_model_submission.csv", index=False)
logger.info("Submission saved!")


# SOMETHING IS WRONG WITH THE MODEL
# LEADERBOARD SCORE: 3.59 xddd
