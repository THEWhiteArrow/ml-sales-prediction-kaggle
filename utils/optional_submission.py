import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


def solve_optional_submission(
    holidays_events: pd.DataFrame,
    oil: pd.DataFrame,
    stores: pd.DataFrame,
    transactions: pd.DataFrame,
    sales: pd.DataFrame,
    query: pd.DataFrame,
) -> pd.DataFrame:
    sales = sales.copy()
    holidays_events.copy()
    query = query.copy()

    sales = sales.set_index(["date", "store_nbr", "family"]).sort_index()
    query = query.set_index(["date", "store_nbr", "family"]).sort_index()
    y = sales.unstack(["store_nbr", "family"]).loc["2017"]  # type: ignore

    fourier = CalendarFourier(freq="ME", order=4)
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True,
    )

    X = dp.in_sample()
    X["NewYear"] = X.index.dayofyear == 1  # type: ignore

    model = LinearRegression(fit_intercept=False).fit(X, y["sales"])
    # y_pred = pd.DataFrame(model.predict(X), index=X.index, columns=y.columns)

    X_query = dp.out_of_sample(steps=16)
    X_query.index.name = "date"
    X_query["NewYear"] = X_query.index.dayofyear == 1  # type: ignore

    y_submit = pd.DataFrame(
        model.predict(X_query), index=X_query.index, columns=y.columns
    )
    y_submit = y_submit.stack(["store_nbr", "family"])
    y_submit = y_submit.join(query.id).reindex(columns=["id", "sales"])

    return y_submit


def solve_optional_customized_submission(
    holidays_events: pd.DataFrame, sales: pd.DataFrame, query: pd.DataFrame
) -> pd.DataFrame:
    sales = sales.copy()
    holidays_events.copy()
    query = query.copy()
    query = query.set_index(["date", "store_nbr", "family"]).sort_index()

    all_families = sales["family"].unique()
    all_stores = sales["store_nbr"].unique()
    cnt = 0

    output = pd.DataFrame({"id": [], "sales": []})

    for store_n in all_stores:
        for family in all_families:
            cnt += 1
            if cnt % 100 == 0:
                print(f"Processing {cnt} of {len(all_families)*len(all_stores)}")

            y = (
                sales[(sales["family"] == family) & (sales["store_nbr"] == store_n)]
                .set_index("date")
                .drop(columns=["store_nbr", "family"])
            )
            y = y.loc["2017"]
            fourier = CalendarFourier(freq="ME", order=4)
            dp = DeterministicProcess(
                index=y.index,
                constant=True,
                order=1,
                seasonal=True,
                additional_terms=[fourier],
                drop=True,
            )
            X = dp.in_sample()
            X["NewYear"] = X.index.dayofyear == 1  # type: ignore
            model = LinearRegression(fit_intercept=False).fit(X, y["sales"])

            X_query = dp.out_of_sample(steps=16)
            X_query.index.name = "date"
            X_query["NewYear"] = X_query.index.dayofyear == 1  # type: ignore
            y_submit = pd.DataFrame(
                model.predict(X_query), index=X_query.index, columns=y.columns
            )
            y_submit["store_nbr"] = store_n
            y_submit["family"] = family
            y_submit = y_submit.set_index(["family", "store_nbr"], append=True)
            y_submit = y_submit.join(query.id).reindex(columns=["id", "sales"])
            output = pd.concat([output, y_submit])

    return output
