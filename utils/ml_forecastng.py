import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from learntools.time_series.utils import (
    make_lags,
    make_leads,
    make_multistep_target,
    plot_multistep,
)

# from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from warnings import simplefilter

simplefilter("ignore")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)


class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None
        self.stack_cols = None

    def fit(self, X_1, X_2, y, stack_cols=None):
        # Train model_1
        self.model_1.fit(X_1, y)

        # Make predictions
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index,
            columns=y.columns,
        )
        # Compute residuals
        y_resid = y - y_fit
        if stack_cols is not None:
            y_resid = y_resid.stack(stack_cols).squeeze()  # wide to long

        # Train model_2 on residuals
        self.model_2.fit(X_2, y_resid)

        # Save column names for predict method
        self.y_columns = y.columns
        self.stack_cols = stack_cols

    def predict(self, X_1, X_2):
        # Predict with model_1
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index,
            columns=self.y_columns,
        )
        if self.stack_cols is not None:
            y_pred = y_pred.stack(self.stack_cols).squeeze()  # wide to long

        # Add model_2 predictions to model_1 predictions
        y_pred += self.model_2.predict(X_2)
        if self.stack_cols is not None:
            y_pred = y_pred.unstack(self.stack_cols)
        return y_pred


def solve_using_ml_forecasting(
    holidays_events: pd.DataFrame,
    oil: pd.DataFrame,
    stores: pd.DataFrame,
    transactions: pd.DataFrame,
    store_sales: pd.DataFrame,
    query: pd.DataFrame,
) -> pd.DataFrame:
    # --- SETUP ---
    store_sales = store_sales.copy()
    query = query.copy()
    holidays_events = holidays_events.copy()
    oil = oil.copy()
    limit = 1
    all_families = store_sales["family"].unique()[:limit]
    all_store_nbrs = store_sales["store_nbr"].unique()[:limit]
    output = pd.DataFrame({"id": [], "sales": []})

    # --- ML FORECASTING ---
    store_sales = (
        store_sales.set_index(["date", "store_nbr", "family"])
        .sort_index()
        .unstack(["store_nbr", "family"])  # type: ignore
    )
    # START = "2016-04-01"
    # END = "2016-05-01"
    START = "2017"
    END = "2017-12-31"

    for family in all_families:
        for store_nbr in all_store_nbrs:
            print(f"Family: {family}, Store: {store_nbr}")
            y = store_sales["sales"][store_nbr][family][START:END].rename("sales").to_frame()  # type: ignore
            onpromotion = store_sales["onpromotion"][store_nbr][family][START:END].rename("onpromotion").to_frame()  # type: ignore
            X_lag = make_lags(y["sales"], lags=4).bfill()  # type: ignore
            X_promo = pd.concat(
                [
                    make_lags(
                        onpromotion["onpromotion"], lags=1, name="onpromotion"
                    ).bfill(),
                    onpromotion,
                    make_leads(
                        onpromotion["onpromotion"], leads=1, name="onpromotion"
                    ).ffill(),
                ],
                axis=1,
            )
            X_holidays = (
                holidays_events[
                    holidays_events["transferred"].eq(False)
                    & holidays_events["locale"].isin(["National", "Regional"])
                ]["type"]
                .cat.remove_unused_categories()
                .cat.add_categories("No Holiday")
                .to_frame()
                .drop_duplicates()
            )
            X_oil = pd.concat(
                [
                    oil["dcoilwtico"],
                    make_lags(oil["dcoilwtico"], lags=2, name="oil"),
                ],
                axis=1,
            )
            X1 = X_lag
            X2 = pd.get_dummies(
                X_promo.join(X_oil)
                .ffill()
                .bfill()
                .join(X_holidays)
                .fillna("No Holiday"),
                drop_first=True,
            )

            y = make_multistep_target(y, steps=16).ffill()  # type: ignore
            print(X1)
            print(X2)
            y, X1 = y.align(X1, join="inner", axis=0)
            y, X2 = y.align(X2, join="inner", axis=0)

            X1_train, X1_test, y_train, y_test = train_test_split(
                X1, y, test_size=0.2, shuffle=False
            )
            X2_train, X2_test, y_train, y_test = train_test_split(
                X2, y, test_size=0.2, shuffle=False
            )

            model = BoostedHybrid(LinearRegression(), XGBRegressor())
            # model = MultiOutputRegressor(XGBRegressor())
            model.fit(X1_train, X2_train, y_train)

            y_fit = pd.DataFrame(
                model.predict(X1_train, X2_train), index=X1_train.index, columns=y.columns  # type: ignore
            )
            y_pred = pd.DataFrame(
                model.predict(X1_test, X2_test), index=X1_test.index, columns=y.columns  # type: ignore
            )

            train_rmse = mean_squared_error(y_train, y_fit, squared=False)
            test_rmse = mean_squared_error(y_test, y_pred, squared=False)
            print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

            palette = dict(palette="husl", n_colors=64)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
            fig.suptitle(f"Family: {family}, Store: {store_nbr}")
            ax1 = store_sales["sales"][store_nbr][family][y_fit.index].plot(
                ax=ax1, **plot_params  # type: ignore
            )
            ax1 = plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
            _ = ax1.legend(["Actual Sales (train)", "Predicted Sales (train)"])
            ax2 = store_sales["sales"][store_nbr][family][y_pred.index].plot(
                ax=ax2, **plot_params  # type: ignore
            )
            ax2 = plot_multistep(y_pred, ax=ax2, palette_kwargs=palette)
            _ = ax2.legend(["Actual Sales (test)", "Predicted Sales (test)"])
            plt.show()

            # --- PREDICT ---
            # TODO: Implement query predictions

    return output
