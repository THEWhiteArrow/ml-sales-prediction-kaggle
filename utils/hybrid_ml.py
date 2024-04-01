from turtle import color
from typing import cast
from cv2 import line
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from learntools.time_series.utils import (
    make_lags,
    make_leads,
    make_multistep_target,
    plot_multistep,
)
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor


def solve_using_hybrid_ml_complex(
    holidays_events: pd.DataFrame,
    oil: pd.DataFrame,
    stores: pd.DataFrame,
    transactions: pd.DataFrame,
    sales: pd.DataFrame,
    query: pd.DataFrame,
) -> pd.DataFrame:

    # --- SETUP ---
    holidays_events = holidays_events.copy()
    oil = oil.copy()
    stores = stores.copy()
    transactions = transactions.copy()
    sales = sales.copy()
    query = query.copy()
    output = pd.DataFrame({"id": [], "sales": []})

    # --- DATA PREPROCESSING ---
    print("Data Preprocessing...")
    YEARS_OF_TRAINING = [2017]
    sales = (
        sales.where(sales["date"].dt.year.isin(YEARS_OF_TRAINING))
        .dropna()
        .set_index(["store_nbr", "family", "date"])
        .sort_index()
    )
    query = query.set_index(["store_nbr", "family", "date"]).sort_index()
    onpromotion = sales["onpromotion"]
    holidays = holidays_events[
        holidays_events["type"].isin(["Holiday", "Additional", "Bridge", "Transfer"])
        & holidays_events["locale"].isin(["National", "Regional"])
        & holidays_events["transferred"].eq(False)
    ]
    holidays["type"] = (
        holidays["type"].replace("Transfer", "Holiday").cat.remove_unused_categories()
    )

    # --- EXPLORATORY DATA ANALYSIS ---

    # --- FEATURE ENGINEERING ---
    print("Feature Engineering...")
    y = sales["sales"]

    X_lag = make_lags(y, lags=4, name="sales").dropna()
    X_promo = pd.concat(
        [
            make_lags(onpromotion, lags=1, name="promo"),
            onpromotion,
            make_leads(onpromotion, leads=1, name="promo"),
        ],
        axis=1,
    )
    # --- NOTICE ---
    # holiday can also be a leading indicator as well as a lagging indicator
    holidays.index = holidays.index.to_timestamp()  # type: ignore
    complete_date_range = pd.date_range(holidays.index.min(), holidays.index.max())
    holidays = holidays[~holidays.index.duplicated(keep="first")]
    holidays = holidays.reindex(complete_date_range)
    holidays["type"] = holidays["type"].cat.add_categories(["No Holiday"])
    holidays["type"] = holidays["type"].fillna("No Holiday")
    holidays.index = holidays.index.to_period("D")
    holidays_time_features = pd.concat(
        [
            make_lags(holidays["type"], lags=1, name="holiday"),
            holidays["type"],
            make_leads(holidays["type"], leads=1, name="holiday"),
        ],
        axis=1,
    ).dropna()
    X_holidays = pd.get_dummies(holidays_time_features, drop_first=True)

    X_lag = X_lag.unstack(["family", "store_nbr"])  # type: ignore
    X_promo = X_promo.unstack(["family", "store_nbr"])  # type: ignore
    print("here")
    # X = (
    #     X_lag.join(X_promo)
    #     .stack(["family", "store_nbr"])
    #     .reset_index()
    #     .join(X_holidays, on="date")
    #     .set_index(["date", "family", "store_nbr"])
    # )
    X = X_lag.join(X_promo).dropna()
    y = make_multistep_target(y, steps=16).dropna().unstack(["family", "store_nbr"])  # type: ignore

    y, X = y.align(X, axis=0, join="inner")
    # --- MODEL ---
    print("Modeling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )

    model = RegressorChain(XGBRegressor()).fit(X_train, y_train)

    # --- NOTICE ---
    # The following code includes some NaN values which need to be removed
    # Should it be stacked or unstacked? -> wide to long maybe?

    # --- EVALUATION ---
    print("Evaluation...")
    y_pred = model.predict(X_test)
    rmsle = mean_squared_log_error(y_test, y_pred) ** 0.5
    print(f"RMSLE: {rmsle}")

    # FAMILY = "PRODUCE"
    # STORE = 1
    # palette = dict(palette="husl", n_colors=64)
    # fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10))

    # --- PREDICTION ---
    print("Prediction...")

    return output


def solve_using_hybrid_ml(
    holidays_events: pd.DataFrame,
    oil: pd.DataFrame,
    stores: pd.DataFrame,
    transactions: pd.DataFrame,
    sales: pd.DataFrame,
    query: pd.DataFrame,
) -> pd.DataFrame:

    # --- SETUP ---
    holidays_events = holidays_events.copy()
    oil = oil.copy()
    stores = stores.copy()
    transactions = transactions.copy()
    sales = sales.copy()
    query = query.copy()
    output = pd.DataFrame({"id": [], "sales": []})

    all_families = sales["family"].unique()[:1]
    all_stores = sales["store_nbr"].unique()[:1]

    holidays = holidays_events[
        holidays_events["type"].isin(["Holiday", "Additional", "Bridge", "Transfer"])
        & holidays_events["locale"].isin(["National", "Regional"])
        & holidays_events["transferred"].eq(False)
    ]
    holidays["type"] = (
        holidays["type"].replace("Transfer", "Holiday").cat.remove_unused_categories()
    )
    holidays.index = holidays.index.to_timestamp()  # type: ignore
    complete_date_range = pd.date_range(holidays.index.min(), holidays.index.max())
    holidays = holidays[~holidays.index.duplicated(keep="first")]
    holidays = holidays.reindex(complete_date_range)
    holidays["type"] = holidays["type"].cat.add_categories(["No Holiday"])
    holidays["type"] = holidays["type"].fillna("No Holiday")
    holidays.index = holidays.index.to_period("D")
    holidays_time_features = pd.concat(
        [
            make_lags(holidays["type"], lags=1, name="holiday"),
            holidays["type"],
            make_leads(holidays["type"], leads=1, name="holiday"),
        ],
        axis=1,
    ).dropna()
    X_holidays = pd.get_dummies(holidays_time_features, drop_first=True)

    for store_nbr in all_stores:
        for family in all_families:

            # --- DATA PREPROCESSING ---
            temp = (
                cast(
                    pd.DataFrame,
                    sales[
                        (sales["store_nbr"] == store_nbr) & (sales["family"] == family)
                    ],
                )
                .set_index("date")
                .loc["2017"]
            )

            # --- FEATURE ENGINEERING ---
            y = temp["sales"].to_frame()
            temp_sales = y.copy()
            onpromotion = temp["onpromotion"]

            X_lag = make_lags(y["sales"], lags=4, name="sales").dropna()
            X_promo = pd.concat(
                [
                    make_lags(onpromotion, lags=1, name="promo"),
                    onpromotion,
                    make_leads(onpromotion, leads=1, name="promo"),
                ],
                axis=1,
            ).dropna()

            X = X_lag.join(X_promo).join(X_holidays).dropna()
            # y = make_multistep_target(y, steps=2).dropna()
            y, X = y.align(X, axis=0, join="inner")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=30, shuffle=False
            )
            # print(X_train)
            # print(y_train)
            print("Modeling...")
            model = RegressorChain(XGBRegressor()).fit(X_train, y_train)

            print("Evaluation...")

            # y_pred = pd.DataFrame(
            #     model.predict(X_test), index=X_test.index, columns=y.columns
            # )
            # y_pred[y_pred < 0] = 0
            # rmsle = mean_squared_log_error(y_test, y_pred) ** 0.5
            # print(f"RMSLE: {rmsle}")

            # ax = y_test.plot(color="black", linestyle="-")
            # y_pred.plot(ax=ax, color="red", linestyle="--")
            # plt.show()
            # ax2 = temp_sales.loc["2017-07"].plot(color="black", linestyle="-")
            # ax2 = plot_multistep(y_pred, ax=ax2)
            # ax2.legend(["Actual", "Predicted"])
            # plt.show()

            # --- PREDICTION ---
            # print("Prediction...")
            # temp_query = cast(
            #     pd.DataFrame,
            #     query[(query["store_nbr"] == store_nbr) & (query["family"] == family)],
            # ).set_index("date")

            # Q_lag = make_lags(temp_query["sales"], lags=4, name="sales").dropna()
            # Q_promo = pd.concat(
            #     [
            #         make_lags(temp_query["onpromotion"], lags=1, name="promo"),
            #         temp_query["onpromotion"],
            #         make_leads(temp_query["onpromotion"], leads=1, name="promo"),
            #     ],
            #     axis=1,
            # ).dropna()
            # Q = Q_lag.join(Q_promo).join(X_holidays).dropna()
            # y_query = pd.DataFrame(model.predict(Q), index=Q.index, columns=y.columns)
            # y_query[y_query < 0] = 0
            # print(y_query)

    return output
