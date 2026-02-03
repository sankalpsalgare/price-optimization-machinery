import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split




def prepare_elasticity_features(df):
    df = df.copy()

    df = df[(df["quantity"] > 0) & (df["avg_price"] > 0)]

    df["log_q"] = np.log(df["quantity"])
    df["log_p"] = np.log(df["avg_price"])

    return df



def add_week_id(df, date_col="date", week_start="MON"):
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
    df = df.copy()

    df["month"] = df[date_col].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_covid_flag(df, date_col="date"):
    df = df.copy()

    df["covid_flag"] = (
        (df[date_col] >= "2020-03-01") &
        (df[date_col] <= "2021-09-30")
    ).astype(int)

    return df


def add_drought_flag(df, date_col="date",drought_years=[2016,2019]):
    df["drought_flag"] = (
        pd.to_datetime(df[date_col])
        .dt.year
        .isin(drought_years)
        .astype(int)
    )
    return df


def weekly_aggregation(df):
    weekly_df = (
        df
        .groupby(
            ["product_name", "week_id"],
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



from sklearn.linear_model import LinearRegression

def estimate_elasticity(df):
    results = []

    feature_cols = [
        "log_p",
        "covid_flag",
        "drought_index",
        "month_sin",
        "month_cos"
    ]

    for product_name, g in df.groupby(["product_name"]):
        if len(g) < 15:
            continue

        X = g[feature_cols]
        y = g["log_q"]

        model = LinearRegression()
        model.fit(X, y)

        results.append({
            "product_name": product_name,
            "price_elasticity": model.coef_[0],
            "n_weeks": len(g)
        })

    return pd.DataFrame(results)


def elasticity_pipeline_OLS_Model(df, date_col="invoice_date"):
    df = add_week_id(df,date_col=date_col)
    df = add_seasonality(df,date_col=date_col)
    df = add_covid_flag(df,date_col=date_col)
    df = add_drought_flag(df, date_col=date_col)

    weekly_df = weekly_aggregation(df)
    weekly_df = prepare_elasticity_features(weekly_df)

    elasticity_df = estimate_elasticity(weekly_df)

    return elasticity_df, weekly_df


def train_test_split_time(df, test_size=0.2):
    max_week = df["week_id"].max()
    split_week = max_week - int(test_size * df["week_id"].nunique())

    train_df = df[df["week_id"] <= split_week]
    test_df = df[df["week_id"] > split_week]

    return train_df, test_df

def add_lagged_features(df, price_lags=[1, 2], demand_lags=[1]):
    df = df.sort_values(["product_name", "week_id"]).copy()

    for lag in price_lags:
        df[f"log_p_lag{lag}"] = (
            df
            .groupby("product_name")["log_p"]
            .shift(lag)
        )

    for lag in demand_lags:
        df[f"log_q_lag{lag}"] = (
            df
            .groupby("product_name")["log_q"]
            .shift(lag)
        )

    return df

def drop_na_lags(df):
    lag_cols = [c for c in df.columns if "lag" in c]
    return df.dropna(subset=lag_cols)


def get_feature_target(df):
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

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_models():
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
        )
    }


def mape(y_true, y_pred):
    y_true = np.exp(y_true)
    y_pred = np.exp(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_models(df):
    results = []

    models = get_models()

    for product_name, g in df.groupby("product_name"):
        if len(g) < 30:
            continue

        train_df, test_df = train_test_split_time(g)

        X_train, y_train = get_feature_target(train_df)
        X_test, y_test = get_feature_target(test_df)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                "product_name": product_name,
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
    df = add_week_id(df,date_col=date_col)
    df = add_seasonality(df,date_col=date_col)
    df = add_covid_flag(df,date_col=date_col)
    df = add_drought_flag(df, date_col=date_col)

    weekly_df = weekly_aggregation(df)
    weekly_df = prepare_elasticity_features(weekly_df)

    comparison_df = evaluate_models(weekly_df)

    return comparison_df, weekly_df


def elasticity_pipeline_with_lags(df, date_col="invoice_date"):
    df = add_week_id(df,date_col="invoice_date")
    df = add_seasonality(df,date_col="invoice_date")
    df = add_covid_flag(df,date_col="invoice_date")
    df = add_drought_flag(df, date_col="invoice_date")

    weekly_df = weekly_aggregation(df)
    weekly_df = prepare_elasticity_features(weekly_df)

    weekly_df = add_lagged_features(weekly_df)
    weekly_df = drop_na_lags(weekly_df)

    comparison_df = evaluate_models(weekly_df)

    return comparison_df, weekly_df
