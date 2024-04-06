from typing import List, cast
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
from utils.helpers import load_files
from utils.logger import logger

logger.name = __name__

# --- DATA PREPARATION ---
logger.info("Preparing data...")
holidays_events, oil, stores, transactions, store_sales, query = load_files()
main_index: List[str] = ["store_nbr", "family", "date"]
secondary_index: List[str] = ["store_nbr", "family"]
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

data["day_of_week"] = data["day_of_week"].astype("category")
data["month"] = data["month"].astype("category")
data["year"] = data["year"].astype("category")


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
    pd.period_range("2016-04-16", periods=100, freq="D")
)
data["earthquake_impact"] = data["earthquake_impact"].astype("category")


# --- NOTICE ---# it is very easy to make a mistake here, so be careful -> depending of how the data is structured you need to do proper lgging and leading
# lagging and leading was issue since it was doing it wrongly while in groups
def make_shift_in_groups(
    df: pd.DataFrame,
    groupby: List[str] = [],
    shift_value: str = "",
    shift: int = 0,
    shift_list: List[int] = [],
    name: str | None = None,
) -> pd.DataFrame:
    shift_list = list(filter(lambda el: el != 0, shift_list))

    if shift == 0 and len(shift_list) == 0:
        raise ValueError(
            "Shift value must be different than 0 or valid shift_list must be provided"
        )

    if shift != 0:
        shift_list.append(shift)

    if name is None:
        name = shift_value

    def create_lagged_columns(group):
        lagged_group = pd.DataFrame(index=group.index)
        for shift in shift_list:

            lagged_group[f"{name}_{'lead' if shift < 0 else 'lag'}_{abs(shift)}"] = (
                group[shift_value].shift(shift)
            )

        return lagged_group

    lagged_df = cast(
        pd.DataFrame,
        df.reset_index(groupby)
        .groupby(groupby, observed=True)
        .apply(create_lagged_columns, include_groups=False),
    )

    return lagged_df


lagged_and_led: List[pd.DataFrame] = [
    make_shift_in_groups(
        data,
        groupby=secondary_index,
        shift_value="sales",
        shift_list=[i for i in range(1, 5)],
    ),
    make_shift_in_groups(
        data,
        groupby=secondary_index,
        shift_value="oil",
        shift_list=[-i for i in range(1, 17)],
    ),
    # make_shift_in_groups(
    #     data,
    #     groupby=secondary_index,
    #     shift_value="onpromotion",
    #     shift_list=[-i for i in range(1, 17)],
    # ),
    # make_shift_in_groups(
    #     data,
    #     groupby=secondary_index,
    #     shift_value="is_holiday",
    #     shift_list=[-i for i in range(1, 17)],
    # ),
]


data_combined = data.join(lagged_and_led)  # type: ignore
data_combined = data_combined.sort_index()
# --- STANDARDIZE AND ENCODE ---
logger.info("Standardizing and encoding...")
logger.warning("Skiping it for now...")

# --- TRAIN-TEST SPLIT ---
logger.info("Splitting data...")

TRAIN_START = "2014-01-01"
TRAIN_END = "2017-06-30"

TEST_START = "2017-07-01"
TEST_END = "2017-07-29"

X = data_combined.drop(columns=["sales", "id"])
y = make_shift_in_groups(
    data_combined,
    groupby=secondary_index,
    shift_value="sales",
    shift_list=[-i for i in range(1, 17)],
)

X_train, y_train = (
    X.loc[:, :, TRAIN_START:TRAIN_END],  # type: ignore
    y.loc[:, :, TRAIN_START:TRAIN_END],  # type: ignore
)
X_test, y_test = (
    X.loc[:, :, TEST_START:TEST_END],  # type: ignore
    y.loc[:, :, TEST_START:TEST_END],  # type: ignore
)

# --- MODEL TRAINING ---
logger.info("Training model...")


# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0, log=True),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, log=True),
#         "gamma": trial.suggest_float("gamma", 0.01, 10.0, log=True),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
#         "min_child_weight": trial.suggest_float(
#             "min_child_weight", 1e-8, 100.0, log=True
#         ),
#     }

#     model = RegressorChain(XGBRegressor(**params))
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     rmse = mean_squared_error(y_test, y_pred)

#     return rmse


# study = optuna.create_study(direction="minimize", study_name="RegressorChain")
# study.optimize(objective, n_trials=50)  # type: ignore

# best_params = study.best_params
# logger.info(f"Best hyperparameters: {best_params}")


# --- PREDICTION ---
logger.info("Making predictions...")
X_data = X.loc[:, :, "2013-02":"2017-07-31"]  # type: ignore
y_data = y.loc[:, :, "2013-02":"2017-07-31"]  # type: ignore
X_query = X.loc[:, :, "2017-08-01":"2017-08-15"]  # type: ignore

legacy_best_params = {
    "n_estimators": 900,
    "max_depth": 12,
    "learning_rate": 0.011343587710019755,
    "subsample": 0.645877958161604,
    "colsample_bytree": 0.6786355037270021,
    "gamma": 0.06258912795337752,
    "reg_alpha": 0.4514809312811976,
    "reg_lambda": 1.2158028885073078e-05,
    "min_child_weight": 4.962702002200526,
}


def make_output(
    df: pd.DataFrame,
) -> pd.DataFrame:

    def create_output_rows(group):
        temp = (
            group.tail(1)
            .rename(columns={f"sales_lead_{i+1}": f"2017-08-{16+i}" for i in range(16)})
            .squeeze()
            .rename("sales")
            .to_frame()
        )
        temp.index.name = "date"
        return temp

    output_df = cast(
        pd.DataFrame,
        df.reset_index(secondary_index)
        .groupby(secondary_index, observed=True)
        .apply(create_output_rows, include_groups=False),
    )

    output_df["sales"] = output_df["sales"].clip(lower=0)
    output_df = output_df.reset_index()
    output_df["date"] = output_df["date"].astype("period[D]")  # type: ignore
    output_df["store_nbr"] = output_df["store_nbr"].astype("str").astype("category")
    output_df["family"] = output_df["family"].astype("category")

    output_df = output_df.set_index(main_index)
    return output_df


total = pd.DataFrame()

for store in unique_stores:
    for family in unique_families:
        X_data_single = X_data.loc[(store, family), :]
        y_data_single = y_data.loc[(store, family), :]
        X_query_single = X_query.loc[(store, family), :]
        model = RegressorChain(XGBRegressor(**legacy_best_params))
        model.fit(X_data_single, y_data_single)
        pred = model.predict(X_query_single)
        pred_df = pd.DataFrame(
            model.predict(X_query), index=X_query.index, columns=y_data.columns
        )
        total = pd.concat([total, make_output(pred_df)])

total_final = query.set_index(main_index).join(total)[["id", "sales"]]
total_final.to_csv("total.csv", index=False)

# model = RegressorChain(XGBRegressor(**legacy_best_params))
# model.fit(X_data, y_data)

# pred_df = pd.DataFrame(
#     model.predict(X_query), index=X_query.index, columns=y_data.columns
# )


# submission = make_output(pred_df)
# submission = query.set_index(main_index).join(submission)[["id", "sales"]]
# submission.to_csv("submission.csv", index=False)

logger.info("Submission file created.")
logger.info("Done.")
