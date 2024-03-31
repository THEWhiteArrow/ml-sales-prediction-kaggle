import os
from typing import Tuple
import pandas as pd
import pickle
from functools import lru_cache


@lru_cache(maxsize=None)
def load_files() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Loads up all given data as Dataframes.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    1. holidays_events
    2. oil
    3. stores
    4. transactions
    5. sales
    6. query
    """
    # --- LOAD FROM CACHE ---
    cache_file = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../cache/cached_data.pkl"
    )
    # If cached file exists, load from it
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print("Loading cached data...")
            cached_data = pickle.load(f)
        return cached_data

    # --- LOAD DATA ---
    path = os.path.join(os.path.dirname(__file__), "../data/")
    holidays_events = pd.read_csv(
        path + "holidays_events.csv",
        dtype={
            "type": "category",
            "locale": "category",
            "locale_name": "category",
            "description": "category",
            "transferred": "bool",
        },
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    holidays_events = holidays_events.set_index("date").to_period("D")

    oil = pd.read_csv(path + "oil.csv", parse_dates=["date"])
    oil = oil.set_index("date").to_period("D")

    stores = pd.read_csv(path + "stores.csv")

    transactions = pd.read_csv(path + "transactions.csv", parse_dates=["date"])
    transactions["date"] = transactions["date"].dt.to_period("D")

    sales = pd.read_csv(
        path + "train.csv",
        usecols=["store_nbr", "family", "date", "sales", "onpromotion"],
        dtype={
            "store_nbr": "category",
            "family": "category",
            "sales": "float32",
            "onpromotion": "uint32",
        },
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    sales["date"] = sales["date"].dt.to_period("D")

    query: pd.DataFrame = pd.read_csv(
        path + "test.csv",
        dtype={
            "store_nbr": "category",
            "family": "category",
            "onpromotion": "uint32",
        },
        parse_dates=["date"],
        infer_datetime_format=True,
    )
    query["date"] = query["date"].dt.to_period("D")

    # --- SAVE TO CACHE ---
    with open(cache_file, "wb") as f:
        print("Caching data...")
        pickle.dump((holidays_events, oil, stores, transactions, sales, query), f)

    return holidays_events, oil, stores, transactions, sales, query


def save_submission(y_submit: pd.DataFrame, filename: str) -> None:
    """
    Saves the submission file.

    Args:
    y_submit (pd.DataFrame): The submission file.
    """
    path = os.path.join(os.path.dirname(__file__), "../output/")
    y_submit.to_csv(path + filename, index=False)
    print(f"Submission saved to {path + filename}")
