import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
def prepare_elasticity_data(
    df,
    group_cols=["product_name", "year", "month"],
):
    """
    Aggregates transaction data to compute
    price-quantity relationships.
    """

    agg_df = (
        df.groupby(group_cols)
        .agg(
            avg_price=("selling_price", "mean"),
            total_qty=("quantity", "sum"),
            revenue=("revenue", "sum")
        )
        .reset_index()
    )

    # Remove zero or negative values (log-safe)
    agg_df = agg_df[
        (agg_df["avg_price"] > 0) &
        (agg_df["total_qty"] > 0)
    ]

    return agg_df


def add_log_features(df):
    df = df.copy()
    df["log_price"] = np.log(df["avg_price"])
    df["log_qty"] = np.log(df["total_qty"])
    return df



def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




def get_elasticity(df):
    # Log transform


    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )
    X_train = np.log(train_df["avg_price"])
    y_train = np.log(train_df["total_qty"])

    X_test = np.log(test_df["avg_price"])
    y_test = np.log(test_df["total_qty"])

    # Add constant
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Fit model
    model = sm.OLS(y_train, X_train).fit()

    # Predictions (back to original scale)
    train_pred = np.exp(model.predict(X_train))
    test_pred = np.exp(model.predict(X_test))

    # Actuals
    y_train_actual = train_df["total_qty"]
    y_test_actual = test_df["total_qty"]

    # MAPEs
    train_mape = mape(y_train_actual, train_pred)
    test_mape = mape(y_test_actual, test_pred)

    # Elasticity = coefficient of log(price)
    elasticity = model.params["avg_price"]

    return {
        "elasticity": elasticity,
        "train_mape": train_mape,
        "test_mape": test_mape,
        "model": model
    }

