import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
import matplotlib.pyplot as plt


def check_trend(average_sales: pd.DataFrame) -> None:
    y = average_sales.copy()
    # --- TREND ---
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10))
    ax0 = y.plot(
        style=".-",
        color="0.75",
        markeredgecolor="0.25",
        markerfacecolor="0.25",
        alpha=0.5,
        ax=ax0,
    )
    moving_average = y.rolling(window=365, center=True, min_periods=183).mean()
    moving_average.plot(ax=ax0, color="red", linestyle="-", figsize=(10, 5))

    dp = DeterministicProcess(
        index=y.index,
        constant=False,
        order=2,
        drop=True,
    )

    X = dp.in_sample()
    X_fore = dp.out_of_sample(steps=365)

    model = LinearRegression()
    model.fit(X, y["sales"])

    y_pred = pd.Series(model.predict(X), index=X.index)
    y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

    ax1 = y.plot(
        alpha=0.5,
        title="Average Sales",
        ylabel="items sold",
        figsize=(10, 5),
        style=".-",
        ax=ax1,
    )
    ax1 = y_pred.plot(ax=ax1, linewidth=2, label="Trend", color=(0.0, 0.9, 0.1, 0.5))
    ax1 = y_fore.plot(
        ax=ax1, linewidth=2, label="Trend Forecast", color=(0.1, 0.9, 0.1, 0.9)
    )
    ax1.legend()
    plt.show()
