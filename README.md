📈 Price Optimization for Irrigation & Machinery Retail
Overview

This project simulates a real-world irrigation and machinery retail business and builds an end-to-end analytics pipeline to understand demand behavior and support pricing and revenue optimization.

The business sells:

PVC pipes and fittings

Irrigation accessories

Motor pumps (high value, low volume)

Sales are influenced by seasonality (pre-monsoon peak), customer type, inflation, and external shocks such as COVID and droughts.
The project mirrors how pricing problems are handled in real analytics teams — starting from data validation and ending with elasticity modeling.

Business Problem

Pricing decisions in agriculture-dependent retail are complex:

Demand varies strongly by season

Different customers respond differently to price changes

Inflation and supply shocks impact margins

Some products tolerate price increases, others do not

Core question:

How does demand respond to price changes, and how can pricing be adjusted to improve revenue and profit without hurting volume?

## Project Highlights
- Synthetic transaction data (2019–2025)
- COVID & drought impact modeling
- Seasonality-aware pricing
- Customer-wise elasticity
- Revenue optimization

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit (planned)

## Structure
```text
price-optimizer/
│
├── data/
│   ├── raw/                # Raw synthetic data
│   ├── processed/          # Cleaned & validated datasets
│
├── notebooks/
│   ├── 01_data_sanity_checks.ipynb
│   ├── 02_time_and_seasonality.ipynb
│   ├── 03_customer_and_product_analysis.ipynb
│   ├── 04_price_dynamics.ipynb
│   ├── 05_elasticity_analysis.ipynb
│
├── src/
│   ├── dataprep/           # Data generation & preprocessing
│   ├── eda/                # Reusable EDA metrics & plots
│   ├── validation/         # Sanity & business rule checks
│   ├── modeling/           # Elasticity & optimization logic
├── dashboards/
├── README.md
└── requirements.txt

Notebooks Description
📓 01_data_sanity_checks.ipynb

Purpose:
Ensure the dataset is logically consistent and business-realistic before analysis.

Key checks:

Missing values and data types

Date range validation (up to June 2025)

Revenue consistency (price × quantity)

Cost vs selling price (including loss-making cases)

Why it matters:
Prevents silent data issues and ensures downstream insights are reliable.

📓 02_time_and_seasonality.ipynb

Purpose:
Understand how demand and revenue change over time.

Key analyses:

Year-wise transaction trends

COVID impact (2019–2022 dip)

Recovery post-2022

Month-wise and season-wise demand

Pre-monsoon peak validation

Drought-related demand reduction

Why it matters:
Seasonality is critical for pricing, inventory planning, and forecasting.

📓 03_customer_and_product_analysis.ipynb

Purpose:
Analyze who buys and what they buy.

Key analyses:

Transaction and revenue share by customer type

Farmers as dominant segment, followed by retailers

SKU-level transaction volume

High-price products with lower frequency

Revenue contribution by category and product

Why it matters:
Enables customer- and product-specific pricing strategies.

📓 04_price_dynamics.ipynb

Purpose:
Study how prices behave over time.

Key analyses:

Average selling price trends (inflation effects)

Monthly price volatility and shocks (every 4–6 months)

Margin pressure and below-cost sales

Declining average prices from 2023 → 2024

Why it matters:
Elasticity and optimization models fail if price behavior is unrealistic.

📓 05_elasticity_analysis.ipynb

Purpose:
Quantify how sensitive demand is to price changes.

Key analyses:

Price vs quantity relationships

Monthly SKU-level aggregation

Log-log regression for elasticity estimation

Elasticity by product and category

Why it matters:
Elasticity is the foundation for price optimization and revenue simulation.


Key Project Strengths

Realistic synthetic data (not toy examples)

Explicit modeling of economic shocks

Strong separation of analysis and logic

Business-first validation of results

Clear progression from EDA to modeling