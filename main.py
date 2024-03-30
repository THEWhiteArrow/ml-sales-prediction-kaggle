from typing import Dict
from utils.helpers import load_files, save_submission
from utils.optional_submission import (
    solve_optional_customized_submission,
    solve_optional_submission,
)
from utils.seasonality import check_seasonality
from utils.time_features import check_time_features
from utils.trend import check_trend


def main(options: Dict[str, bool]):
    options = {
        "trend": True,
        "seasonality": True,
        "optional_submission": True,
        "optional_customized_submission": True,
        "time_features": True,
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
        output = solve_optional_customized_submission(holidays_events, sales, query)
        save_submission(output, "optional_customized_submission.csv")

    # --- TIME FEATURES ---
    if options["time_features"]:
        check_time_features(sales=sales)


if __name__ == "__main__":
    print("Starting...")
    main(
        options={
            "trend": False,
            "seasonality": False,
            "optional_submission": False,
            "optional_customized_submission": False,
            "time_features": True,
        }
    )
