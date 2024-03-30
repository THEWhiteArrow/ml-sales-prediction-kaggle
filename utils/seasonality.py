import pandas as pd
from sklearn.linear_model import LinearRegression
from learntools.time_series.utils import seasonal_plot, plot_periodogram
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import matplotlib.pyplot as plt


def check_seasonality(df: pd.DataFrame, holidays_events: pd.DataFrame) -> None:
    # --- SEASONALITY ---
    X = df.copy()
    y = df.copy()
    holidays_events = holidays_events.copy()

    print("next", flush=True)
    week = pd.Series(X.index.week, index=X.index)  # type: ignore
    year = pd.Series(X.index.year, index=X.index)  # type: ignore
    day = pd.Series(X.index.dayofweek, index=X.index)  # type: ignore
    dayofyear = pd.Series(X.index.dayofyear, index=X.index)  # type: ignore

    X = pd.concat([X, week, year, day, dayofyear], axis=1)
    X.columns = ["sales", "week", "year", "day", "dayofyear"]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 10))

    seasonal_plot(X, y="sales", period="week", freq="day", ax=ax0)
    seasonal_plot(X, y="sales", period="year", freq="dayofyear", ax=ax1)
    plot_periodogram(X["sales"], ax=ax2)

    fourier = CalendarFourier(freq="ME", order=4)
    dp = DeterministicProcess(
        index=y.index,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True,
    )

    X2 = dp.in_sample()
    X3 = dp.out_of_sample(20)
    model = LinearRegression(fit_intercept=False).fit(X2, y["sales"])
    y_pred = pd.Series(model.predict(X2), index=X2.index, name="Fitted")
    y_fore = pd.Series(model.predict(X3), index=X3.index, name="Forecast")
    ax3 = y.plot(ax=ax3, alpha=0.5, title="Average Sales", ylabel="Items sold")
    ax3 = y_pred.plot(ax=ax3, label="Seasonal")
    ax3 = y_fore.plot(ax=ax3, label="Forecast", color=(0.1, 0.9, 0.1, 0.9))
    plt.show()

    y_deseason = y["sales"] - y_pred
    fig, (ax4, ax5) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))
    ax4 = plot_periodogram(y["sales"], ax=ax4)
    ax4.set_title("Product Sales Frequency Components")
    ax5 = plot_periodogram(y_deseason, ax=ax5)
    ax5.set_title("Deseasonalized")
    plt.show()

    holidays = (
        holidays_events.query("locale in ['National', 'Regional']")
        .loc["2017":"2017-08-15", ["description"]]
        .assign(description=lambda x: x.description.cat.remove_unused_categories())
    )

    X_holidays = pd.get_dummies(holidays)
    X4 = X2.join(X_holidays, on="date", how="left").fillna(0.0)

    model4 = LinearRegression(fit_intercept=False).fit(X4, y["sales"])
    y_pred4 = pd.Series(model4.predict(X4), index=X4.index, name="FittedWithHolidays")
    ax6 = y["sales"].plot(alpha=0.5, title="Average Sales", ylabel="items sold")
    ax6 = y_pred4.plot(ax=ax6, label="Seasonal")
    ax6.legend()
    plt.show()
