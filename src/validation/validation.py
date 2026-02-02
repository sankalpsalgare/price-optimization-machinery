import pandas as pd


# =========================
# BASIC DATA CHECKS
# =========================
def dataset_shape(df):
    """
    Returns the shape of the dataset.

    Parameters:
    df (pd.DataFrame): Input dataset

    Returns:
    dict: {
        "rows": int,    # total number of rows
        "columns": int  # total number of columns
    }
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }


def date_range_check(df):
    """
    Checks the date range covered by the dataset.

    Required Columns:
    - invoice_date (datetime)

    Returns:
    dict: {
        "min_date": earliest transaction date,
        "max_date": latest transaction date
    }
    """
    return {
        "min_date": df["invoice_date"].min(),
        "max_date": df["invoice_date"].max()
    }


# =========================
# CUSTOMER VALIDATION
# =========================
def customer_transaction_share(df):
    """
    Calculates transaction count distribution by customer type.

    Required Columns:
    - customer_type

    Returns:
    pd.Series: Transaction counts per customer type (descending order)
    """
    return (
        df.groupby("customer_type")
        .size()
        .sort_values(ascending=False)
    )


def customer_revenue_share(df):
    """
    Calculates total revenue contribution by customer type.

    Required Columns:
    - customer_type
    - revenue

    Returns:
    pd.Series: Total revenue per customer type (descending order)
    """
    return (
        df.groupby("customer_type")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

def product_revenue_share(df):
    """
    Calculates total revenue contribution by customer type.

    Required Columns:
    - customer_type
    - revenue

    Returns:
    pd.Series: Total revenue per customer type (descending order)
    """
    return (
        df.groupby('product_name')['revenue']
        .sum()
        .sort_values(ascending=False)
    )

# =========================
# TIME SERIES VALIDATION
# =========================
def yearly_transactions(df):
    """
    Computes number of transactions per year.

    Required Columns:
    - year

    Returns:
    pd.Series: Transaction count indexed by year
    """
    return df.groupby("year").size()


def monthly_transactions(df):
    """
    Computes number of transactions per month.

    Required Columns:
    - month

    Returns:
    pd.Series: Transaction count indexed by month
    """
    return df.groupby("month").size()


def category_yearly_revenue(df):
    """
    Calculates yearly revenue per product category.

    Required Columns:
    - year
    - product_category
    - revenue

    Returns:
    pd.DataFrame with columns:
    - year
    - product_category
    - revenue
    """
    return (
        df.groupby(["year", "product_category"])["revenue"]
        .sum()
        .reset_index()
    )


# =========================
# COVID & SHOCK VALIDATION
# =========================
def covid_period_impact(df):
    """
    Analyzes transaction volume during COVID period (2020–2022).

    Required Columns:
    - year

    Returns:
    pd.Series: Transaction counts per COVID year
    """
    covid_years = df[df["year"].between(2020, 2022)]
    return covid_years.groupby("year").size()


def drought_year_impact(df, drought_years):
    """
    Measures revenue impact during specified drought years.

    Parameters:
    drought_years (list): List of drought-affected years

    Required Columns:
    - year
    - revenue

    Returns:
    pd.Series: Total revenue per drought year
    """
    drought_df = df[df["year"].isin(drought_years)]
    return drought_df.groupby("year")["revenue"].sum()


# =========================
# PRICE & COST VALIDATION
# =========================
def price_below_cost_cases(df):
    """
    Identifies cases where selling price is below unit cost.

    Required Columns:
    - selling_price
    - unit_cost

    Returns:
    dict: {
        "count": number of loss-making transactions,
        "percentage": percentage of total transactions
    }
    """
    below_cost = df[df["selling_price"] < df["unit_cost"]]
    return {
        "count": below_cost.shape[0],
        "percentage": round(len(below_cost) / len(df) * 100, 2)
    }


def avg_price_trend(df):
    """
    Computes average selling price trend over years.

    Required Columns:
    - year
    - selling_price

    Returns:
    pd.Series: Average selling price per year
    """
    return (
        df.groupby("year")["selling_price"]
        .mean()
        .round(2)
    )


def price_volatility_by_month(df):
    """
    Measures monthly price volatility using standard deviation.

    Required Columns:
    - year
    - month
    - selling_price

    Returns:
    pd.DataFrame with columns:
    - year
    - month
    - price_std
    """
    return (
        df.groupby(["year", "month"])["selling_price"]
        .std()
        .reset_index(name="price_std")
    )


# =========================
# PRODUCT-LEVEL VALIDATION
# =========================
def product_transaction_distribution(df):
    """
    Computes transaction count per product.

    Required Columns:
    - product_name

    Returns:
    pd.Series: Transaction counts per product (descending order)
    """
    return (
        df.groupby("product_name")
        .size()
        .sort_values(ascending=False)
    )


def high_price_low_volume_check(df):
    """
    Identifies products with high average price but low transaction volume.

    Required Columns:
    - product_name
    - selling_price

    Returns:
    pd.DataFrame with columns:
    - product_name
    - avg_price
    - transactions
    """
    summary = (
        df.groupby("product_name")
        .agg(
            avg_price=("selling_price", "mean"),
            transactions=("product_name", "count")
        )
        .reset_index()
    )
    return summary.sort_values(by="avg_price", ascending=False)


def category_revenue_share(df):
    """
    Calculates revenue share by product category.

    Required Columns:
    - product_category
    - revenue

    Returns:
    pd.Series: Total revenue per category (ascending order)
    """
    return (
        df.groupby("product_category")["revenue"]
        .sum()
        .sort_values(ascending=True)
    )
