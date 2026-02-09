import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

import re

def create_product_id(
    df,
    product_name_col="product_name",
    specification_col="specification",
    product_id_col="product_id"
):
    """
    Create a unique and stable product identifier by concatenating
    product name and specification.

    This function standardizes text by:
    - lowercasing
    - trimming whitespace
    - replacing non-alphanumeric characters with underscores

    Final format:
        product_name__specification

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing product name and specification.
    product_name_col : str, default="product_name"
        Column name representing the product name.
    specification_col : str, default="specification"
        Column name representing the product specification.
    product_id_col : str, default="product_id"
        Name of the output product_id column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new standardized product_id column.

    Examples
    --------
    product_name = "PVC Pipe"
    specification = "1 inch – Heavy Duty"

    product_id:
    pvc_pipe__1_inch_heavy_duty
    """
    df = df.copy()

    def _clean_text(x):
        x = str(x).lower().strip()
        x = re.sub(r"[^a-z0-9]+", "_", x)
        return x.strip("_")

    df[product_id_col] = (
        df[product_name_col].apply(_clean_text)
        + "__"
        + df[specification_col].apply(_clean_text)
    )

    return df


def add_week_id(df, date_col="date", week_start="MON"):
    """
    Create a continuous, monotonically increasing week identifier
    across multiple years.

    The week_id does NOT reset every year. It is computed relative
    to the earliest week in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a date column.
    date_col : str, default="date"
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - week_start_date
        - week_id
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Align each date to start of its week
    df["week_start_date"] = (
        df[date_col]
        .dt.to_period(f"W-{week_start}")
        .dt.start_time
    )

    # Reference week (earliest week in data)
    ref_week = df["week_start_date"].min()

    # Continuous week id (starting from 1)
    df["week_id"] = (
        (df["week_start_date"] - ref_week)
        .dt.days // 7
    ) + 1

    return df


def add_seasonality(df, date_col="date"):
    """
    Add cyclical seasonality features using month-of-year encoding.

    Uses sine and cosine transformations to avoid artificial
    discontinuities between December and January.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a date column.
    date_col : str, default="date"
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - month_sin
        - month_cos
    """
    df = df.copy()

    df["month"] = df[date_col].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_covid_flag(df, date_col="date"):
    """
    Add a binary COVID-19 impact indicator.

    Flags periods affected by COVID disruptions to control
    for structural demand shocks unrelated to price.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a date column.
    date_col : str, default="date"
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added column:
        - covid_flag (0/1)
    """
    df = df.copy()

    df["covid_flag"] = (
        (df[date_col] >= "2020-03-01") &
        (df[date_col] <= "2021-09-30")
    ).astype(int)

    return df


def add_drought_flag(df, date_col="date",drought_years=[2016,2019]):
    """
    Merge weekly drought severity information into the main dataset.

    Assumes drought_df is already aggregated at week level and
    represents global or business-wide drought impact.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataframe containing week_id.
    drought_df : pd.DataFrame
        DataFrame with columns:
        - week_id
        - drought_index

    Returns
    -------
    pd.DataFrame
        DataFrame with added column:
        - drought_index
    """
    df["drought_flag"] = (
        pd.to_datetime(df[date_col])
        .dt.year
        .isin(drought_years)
        .astype(int)
    )
    return df


def weekly_aggregation(df):
    """
    Aggregate transactional data to weekly, product-level granularity.

    Aggregation logic:
    - quantity: sum
    - revenue: sum
    - price: mean
    - covid_flag: max (any disruption during the week)
    - drought_index: mean
    - seasonality terms: mean

    Parameters
    ----------
    df : pd.DataFrame
        Daily or transactional sales data.

    Returns
    -------
    pd.DataFrame
        Weekly aggregated dataset per product.
    """
    weekly_df = (
        df
        .groupby(
            ["product_id", "week_id"],
            as_index=False
        )
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            avg_price=("selling_price", "mean"),

            # Exogenous variables
            covid_flag=("covid_flag", "max"),
            drought_index=("drought_flag", "mean"),
            month_sin=("month_sin", "mean"),
            month_cos=("month_cos", "mean")
        )
    )

    return weekly_df

def prepare_elasticity_features(df):
    """
    Prepare core log-log variables required for elasticity modeling.

    Filters invalid observations and computes:
    - log(quantity)
    - log(price)

    Parameters
    ----------
    df : pd.DataFrame
        Weekly aggregated data.

    Returns
    -------
    pd.DataFrame
        DataFrame with log-transformed variables.
    """
    df = df.copy()

    df = df[(df["quantity"] > 0) & (df["avg_price"] > 0)]

    df["log_q"] = np.log(df["quantity"])
    df["log_p"] = np.log(df["avg_price"])

    return df

def add_lagged_features(df, price_lags=[1, 2], demand_lags=[1]):
    """
    Add lagged price and demand variables for dynamic elasticity effects.

    Lagged features capture delayed customer response, inventory effects,
    and temporal autocorrelation in demand.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly elasticity-ready dataset.
    price_lags : tuple of int, default=(1, 2)
        Lags (in weeks) for price.
    demand_lags : tuple of int, default=(1,)
        Lags (in weeks) for demand.

    Returns
    -------
    pd.DataFrame
        DataFrame with added lagged features.
    """
    df = df.sort_values(["product_id", "week_id"]).copy()

    for lag in price_lags:
        df[f"log_p_lag{lag}"] = (
            df
            .groupby("product_id")["log_p"]
            .shift(lag)
        )

    for lag in demand_lags:
        df[f"log_q_lag{lag}"] = (
            df
            .groupby("product_id")["log_q"]
            .shift(lag)
        )

    return df

def drop_na_lags(df):
    """
    Drop rows with missing lagged values.

    Necessary because initial weeks per product
    do not have sufficient history.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing lagged features.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame without NA lag rows.
    """
    lag_cols = [c for c in df.columns if "lag" in c]
    return df.dropna(subset=lag_cols)

def train_test_split_time(df, test_size=0.2):
    """
    Perform a time-aware train/test split based on week_id.

    Ensures no future information leaks into training data.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly dataset for a single product.
    test_size : float, default=0.2
        Fraction of data reserved for testing.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Train and test dataframes.
    """
    max_week = df["week_id"].max()
    split_week = max_week - int(test_size * df["week_id"].nunique())

    train_df = df[df["week_id"] <= split_week]
    test_df = df[df["week_id"] > split_week]

    return train_df, test_df

def get_feature_target(df):
    """
    Construct feature matrix X and target vector y
    for elasticity modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly elasticity dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable (log demand).
    """
    X = df[
        [
            "log_p",
            "log_p_lag1",
            "log_p_lag2",
            "log_q_lag1",
            "covid_flag",
            "drought_index",
            "month_sin",
            "month_cos"
        ]
    ]
    y = df["log_q"]

    return X, y

def mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE)
    on the original demand scale.

    Parameters
    ----------
    y_true : array-like
        True log-demand values.
    y_pred : array-like
        Predicted log-demand values.

    Returns
    -------
    float
        MAPE value in percentage.
    """
    y_true = np.exp(y_true)
    y_pred = np.exp(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_models():
    """
    Define a registry of models for elasticity comparison.

    Includes:
    - Linear (interpretable)
    - Regularized linear (stable)
    - Tree-based (predictive benchmark)

    Returns
    -------
    dict
        Dictionary mapping model names to sklearn estimators.
    """
    return {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
    }



def evaluate_models(df):
    """
    Train, test, and compare multiple models per product.

    Evaluation is performed using time-aware splitting
    and MAPE on test data.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly elasticity dataset with lagged features.

    Returns
    -------
    pd.DataFrame
        Model comparison results including MAPE and elasticity.
    """
    results = []

    models = get_models()

    for product_id, g in df.groupby("product_id"):
        if len(g) < 30:
            continue

        train_df, test_df = train_test_split_time(g)

        X_train, y_train = get_feature_target(train_df)
        X_test, y_test = get_feature_target(test_df)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "product_id": product_id,
                "model": model_name,
                "test_mape": mape(y_test, y_pred),
                "n_train_weeks": len(train_df),
                "n_test_weeks": len(test_df),

                # Only meaningful for linear models
                "price_elasticity": (
                    model.coef_[0]
                    if hasattr(model, "coef_")
                    else None
                )
            })

    return pd.DataFrame(results)


def elasticity_pipeline_with_models(df, date_col="invoice_date"):
    """
    End-to-end pipeline for estimating and evaluating
    price elasticities.

    Steps:
    - Feature engineering
    - Weekly aggregation
    - Lagged effects
    - Model comparison with time-aware validation

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactional sales data.
    drought_df : pd.DataFrame
        Weekly drought severity data.

    Returns
    -------
    results_df : pd.DataFrame
        Model comparison results with MAPE and elasticities.
    weekly_df : pd.DataFrame
        Final modeling dataset.
    """
    df=create_product_id(df)
    df = add_week_id(df,date_col=date_col)
    df = add_seasonality(df,date_col=date_col)
    df = add_covid_flag(df,date_col=date_col)
    df = add_drought_flag(df, date_col=date_col)

    weekly_df = weekly_aggregation(df)
    weekly_df = prepare_elasticity_features(weekly_df)
    weekly_df = add_lagged_features(weekly_df)
    weekly_df = drop_na_lags(weekly_df)

    comparison_df = evaluate_models(weekly_df)

    return comparison_df, weekly_df

