from typing import Dict
from utils.helpers import load_files, save_submission
from utils.hybrid_ml import solve_using_hybrid_ml
from utils.ml_forecastng import solve_using_ml_forecasting
from utils.optional_submission import (
    solve_optional_customized_submission,
    solve_optional_submission,
)
from utils.seasonality import check_seasonality
from utils.time_features import check_time_features
from utils.trend import check_trend


def main(options: Dict[str, bool | str | None]):
    options = {
        "trend": True,
        "seasonality": True,
        "optional_submission": True,
        "optional_customized_submission": True,
        "time_features": True,
        "hybrid_ml": True,
        "ml_forecasting": True,
        "family": None,
        "store_nbr": None,
        **options,
    }

    # --- SETUP ---
    holidays_events, oil, stores, transactions, sales, query = load_files()

    # --- EXPLORATORY DATA ANALYSIS ---
    if options["trend"]:
        check_trend(
            sales.copy()
            .set_index(["store_nbr", "family", "date"])
            .sort_index()
            .groupby("date")
            .mean()
        )  # type: ignore

    if options["seasonality"]:
        check_seasonality(sales.copy().set_index(["store_nbr", "family", "date"]).sort_index().groupby("date").mean().loc["2017"], holidays_events=holidays_events)  # type: ignore

    # --- OPTIONAL SUBMISSION ---
    if options["optional_submission"]:
        output = solve_optional_submission(
            holidays_events, oil, stores, transactions, sales, query
        )
        save_submission(output, "optional_submission.csv")

    if options["optional_customized_submission"]:
        output = solve_optional_customized_submission(
            holidays_events, sales, query, options["family"], options["store_nbr"]  # type: ignore
        )
        save_submission(output, "optional_customized_submission.csv")

    # --- TIME FEATURES ---
    if options["time_features"]:
        check_time_features(sales=sales)

    # --- HYBRID MODEL ---
    if options["hybrid_ml"]:
        output = solve_using_hybrid_ml(
            holidays_events=holidays_events,
            oil=oil,
            stores=stores,
            transactions=transactions,
            sales=sales,
            query=query,
        )
        # save_submission(output, "hybrid_submission.py")

    # --- ML FORECASTING ---
    if options["ml_forecasting"]:
        output = solve_using_ml_forecasting(
            holidays_events=holidays_events,
            oil=oil,
            stores=stores,
            transactions=transactions,
            store_sales=sales,
            query=query,
        )


if __name__ == "__main__":
    print("Starting...")
    main(
        options={
            "trend": False,
            "seasonality": False,
            "optional_submission": False,
            "optional_customized_submission": False,
            "time_features": False,
            "hybrid_ml": False,
            "ml_forecasting": True,
            # "family": "AUTOMOTIVE",
            # "store_nbr": "1",
        }
    )
