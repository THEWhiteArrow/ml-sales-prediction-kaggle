import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from learntools.time_series.utils import plot_lags, make_lags, make_leads


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

    # --- DATA PREPROCESSING ---
    print("Data Preprocessing...")
    sales = sales.set_index(["store_nbr", "family", "date"]).sort_index()
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

    X_lag = make_lags(y, lags=4)
    X_promo = pd.concat(
        [make_lags(onpromotion, lags=1), onpromotion, make_leads(onpromotion, leads=1)],
        axis=1,
    )
    # --- NOTICE ---
    # holiday can also be a leading indicator as well as a lagging indicator
    holidays.index = holidays.index.to_timestamp()
    complete_date_range = pd.date_range(holidays.index.min(), holidays.index.max())
    holidays = holidays[~holidays.index.duplicated(keep="first")]
    holidays = holidays.reindex(complete_date_range)
    holidays["type"] = holidays["type"].cat.add_categories(["No Holiday"])
    holidays["type"] = holidays["type"].fillna("No Holiday")
    holidays_time_features = pd.concat(
        [
            make_lags(holidays["type"], lags=1),
            holidays["type"],
            make_leads(holidays["type"], leads=1),
        ],
        axis=1,
    ).dropna()
    X_holidays_dummies = pd.get_dummies(holidays_time_features, drop_first=True)

    # cannot use pd.concat because of the different indices
    # X = pd.concat([X_lag, X_promo, X_holidays_dummies], axis=1).dropna()

    # --- MODEL ---
    print("Modeling...")

    # --- VERIFICATION ---
    print("Verification...")

    # --- PREDICTION ---
    print("Prediction...")

    return output
