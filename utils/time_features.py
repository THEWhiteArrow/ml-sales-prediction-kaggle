from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.graphics.tsaplots import plot_pacf
from learntools.time_series.utils import plot_lags, make_lags, make_leads


def check_time_features(sales: pd.DataFrame) -> None:
    store_sales = sales.copy()
    print("Checking time features...")

    store_sales = store_sales.set_index(["store_nbr", "family", "date"]).sort_index()
    print(store_sales.head())
    family_sales = (
        store_sales.groupby(["family", "date"])
        .mean()
        .unstack("family")
        .loc["2017", ["sales", "onpromotion"]]
    )  # type: ignore

    supply_sales = family_sales.loc(axis=1)[:, "SCHOOL AND OFFICE SUPPLIES"]
    y = supply_sales.loc[:, "sales"].squeeze()  # type: ignore

    fourier = CalendarFourier(freq="ME", order=4)
    dp = DeterministicProcess(
        constant=True,
        index=y.index,
        order=1,
        seasonal=True,
        drop=True,
        additional_terms=[fourier],
    )
    X_time = dp.in_sample()
    X_time["NewYearsDay"] = X_time.index.dayofyear == 1  # type: ignore
    model = LinearRegression(fit_intercept=False).fit(X_time, y)

    y_deseason = y - model.predict(X_time)
    y_deseason.name = "sales_deseasoned"

    fig, (ax0, ax1, ax2) = plt.subplots(
        3,
        1,
        figsize=(10, 10),
    )
    ax0 = y_deseason.plot(
        ax=ax0, title="Sales of School and Office Supplies (deseasonalized)"
    )
    y_moving_average = y.rolling(7, center=True).mean()
    ax1 = y_moving_average.plot(ax=ax1, title="Seven-Day Moving Average")

    plot_pacf(x=y_deseason, ax=ax2, lags=8)
    plot_lags(x=y_deseason, lags=8, nrows=1)

    onpromotion = supply_sales.loc[:, "onpromotion"].squeeze().rename("onpromotion")
    plot_lags(
        x=onpromotion.loc[onpromotion > 1],
        y=y_deseason.loc[onpromotion > 1],
        lags=3,
        leads=3,
        nrows=2,
    )
    plt.show()

    X_lags = make_lags(y_deseason, lags=1)
    X_promo = pd.concat(
        [make_lags(onpromotion, lags=1), onpromotion, make_leads(onpromotion, leads=1)],
        axis=1,
    )
    y_lag = supply_sales.loc[:, "sales"].shift(1)
    X_std = y_lag.rolling(7).std()

    X = pd.concat([X_time, X_lags, X_promo, X_std], axis=1).dropna()
    # --- NOTICE ---
    """
    Code below makes sure that both dataframes have the same indexes.
    """
    y, X = y.align(X, join="inner")

    # --- VALIDATE ---
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=30, shuffle=False
    )

    model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    y_fit = pd.Series(model.predict(X_train), index=X_train.index).clip(0.0)
    y_pred = pd.Series(model.predict(X_valid), index=X_valid.index).clip(0.0)

    rmsle_train = mean_squared_log_error(y_train, y_fit) ** 0.5
    rmsle_valid = mean_squared_log_error(y_valid, y_pred) ** 0.5
    print(f"Training RMSLE: {rmsle_train:.5f}")
    print(f"Validation RMSLE: {rmsle_valid:.5f}")

    ax = y.plot(alpha=0.5, title="Average Sales", ylabel="items sold")
    ax = y_fit.plot(ax=ax, label="Fitted", color="C0")
    ax = y_pred.plot(ax=ax, label="Forecast", color="C3")
    ax.legend()
    plt.show()
    print("Checked time features...")
