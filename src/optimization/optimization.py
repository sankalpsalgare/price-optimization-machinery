import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
import math
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import *
from elasticity.elasticity import *



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

def optimization_dataprep(
        df,
        date_col='date',
        n_weeks: int=10,
        product_name_col="product_name",
        specification_col="specification",
        product_id_col="product_id"
):
    df = df.copy()
    df=create_product_id(df,product_id_col=product_id_col,product_name_col=product_name_col,specification_col=specification_col)
    df=add_week_id(df,date_col=date_col)
    last_weeks = (
        df["week_id"]
        .drop_duplicates()
        .sort_values()
        .tail(n_weeks)
    )
    df = df[df["week_id"].isin(last_weeks)]
    agg_df = (
        df
        .groupby(["product_id","product_category"], as_index=False)
        .agg(
            base_price=("selling_price", "mean"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            discount=("discount_pct", "sum"),
            units_sold=("quantity", "sum"),
        )
    )
    agg_df["base_price"]=agg_df["base_price"].astype(int)
    agg_df["revenue"]=agg_df["revenue"].astype(int)
    agg_df["profit"]=agg_df["profit"].astype(int)
    agg_df["discount"]=agg_df["discount"].astype(int)
    return agg_df


def simulate_revenue(base_price, base_units, elasticity, new_price):
    """
    Simulates units sold and revenue using constant elasticity demand.
    """
    new_units = base_units * (new_price / base_price) ** elasticity
    new_revenue = new_price * new_units
    return new_units, new_revenue


def get_max_price_from_config(base_price, category, config):
    """
    Computes max allowed price using config rules.
    """
    rules = config[category.lower()]

    max_abs = rules["max_abs_increase"]
    max_pct = rules["max_pct_increase"] * base_price

    max_increase = min(max_abs, max_pct)
    return base_price + max_increase



def optimize_revenue_single_product(
    base_price,
    base_units,
    elasticity,
    category,
    config,
    n_iterations=DEFAULT_ITERATIONS
):
    """
    Revenue maximization using config-driven constraints.
    """

    rules = config[category.lower()]
    max_unit_drop = rules["max_unit_drop"]

    max_price = get_max_price_from_config(
        base_price, category, config
    )

    base_revenue = base_price * base_units

    best_price = base_price
    best_units = base_units
    best_revenue = base_revenue

    for _ in range(n_iterations):
        candidate_price = np.random.uniform(base_price, max_price)

        units, revenue = simulate_revenue(
            base_price, base_units, elasticity, candidate_price
        )

        # Volume constraint
        if units < base_units * (1 - max_unit_drop):
            continue

        if revenue > best_revenue:
            best_price = round(candidate_price)
            best_units = units
            best_revenue = revenue

    return {
        "optimized_price": best_price,
        "optimized_units": best_units,
        "optimized_revenue": best_revenue,
        "revenue_change_pct": 
            (best_revenue - base_revenue) / base_revenue
    }


def optimize_revenue_dataframe(
    df,
    config=PRICING_CONSTRAINTS,
    n_iterations=DEFAULT_ITERATIONS
):
    """
    Runs config-driven revenue optimization for all products.
    """

    results = []

    for _, row in df.iterrows():
        result = optimize_revenue_single_product(
            base_price=row["base_price"],
            base_units=row["units_sold"],
            elasticity=row["price_elasticity"],
            category=row["product_category"],
            config=config,
            n_iterations=n_iterations
        )

        results.append({
            "product_id": row["product_id"],
            "category": row["product_category"],
            "base_price": row["base_price"],
            "optimized_price": result["optimized_price"],
            "base_units": row["units_sold"],
            "optimized_units": result["optimized_units"],
            "base_revenue": row["revenue"],
            "optimized_revenue": result["optimized_revenue"],
            "revenue_change_pct": result["revenue_change_pct"]
        })

    return pd.DataFrame(results)
