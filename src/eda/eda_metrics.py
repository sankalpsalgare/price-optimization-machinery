import pandas as pd


def revenue_by_category(df):
    """
    Calculates total revenue by product category.

    Required Columns:
    - category
    - revenue

    Returns:
    pd.Series: Total revenue per category (descending order)
    """
    return (
        df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )


def revenue_by_season(df):
    """
    Calculates total revenue by season.

    Required Columns:
    - season
    - revenue

    Returns:
    pd.Series: Total revenue per season (descending order)
    """
    return (
        df.groupby("season")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )


def motor_pump_monthly_revenue(df):
    """
    Calculates monthly revenue for Motor category products.

    Required Columns:
    - product_category
    - month
    - revenue

    Returns:
    pd.Series: Monthly revenue for Motor category (ascending order)
    """
    motor_df = df[df["product_category"] == "Motor"]
    return (
        motor_df.groupby("month")["revenue"]
        .sum()
        .sort_values(ascending=True)
    )


def top_products_by_revenue(df, n=10):
    """
    Identifies top N products by total revenue.

    Parameters:
    n (int): Number of top products to return (default = 10)

    Required Columns:
    - product_name
    - revenue

    Returns:
    pd.Series: Top N products ranked by total revenue
    """
    return (
        df.groupby("product_name")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )
