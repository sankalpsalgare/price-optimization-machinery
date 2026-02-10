# 📈 Price Optimization & Revenue Maximization Pipeline

## Overview

This project implements an **end-to-end price optimization system** designed to maximize **revenue and profit** using **historical sales data** and **product-level price elasticity modeling**.

The pipeline takes raw transactional data, cleans and aggregates it into a **week × product** format, estimates **own-price elasticities**, and then applies **constrained optimization** to recommend optimal prices under realistic business rules such as:

* Maximum absolute price increase
* Maximum percentage price change
* Optional demand drop constraints
* Price rounding rules

The system is modular, configurable, and built to be extended (e.g., category-level rules, profit optimization, scenario testing).

---

## Business Objective

The core objective is to answer:

> *“Given how sensitive demand is to price changes, what is the **best price** for each product that maximizes revenue (or profit) **without violating business constraints**?”*

This solution is especially suited for:

* Retail & FMCG pricing
* Limited product catalogs
* Scenarios where **self-elasticity** is more important than cross-elasticity

---

## Key Features

* ✅ Product-level **price elasticity estimation**
* ✅ Weekly aggregation (price averaged, volumes summed)
* ✅ External factor handling (seasonality, COVID, drought, etc.)
* ✅ Revenue optimization with **hard price constraints**
* ✅ Config-driven rules (easy to tweak)
* ✅ Clean, reproducible pipeline
* ✅ Optimized price rounding

---

## Project Structure

```
price_optimizer/
│
├── data/
│   ├── raw/                  # Raw transactional data
│   ├── processed/            # Cleaned & aggregated datasets
│
├── elasticity/
│   ├── elasticity_utils.py   # Elasticity calculation functions
│   ├── models.py             # Regression / elasticity models
│
├── optimizer/
│   ├── dataprep.py           # Optimization-ready dataset creation
│   ├── revenue_optimizer.py  # Core optimization logic
│   ├── rounding.py           # Price rounding rules
│
├── config.py                 # All business constraints & configs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_elasticity_analysis.ipynb
│   ├── 03_price_optimization.ipynb
│
├── requirements.txt
└── README.md
```

---

## Data Assumptions

### Input Data (Transactional Level)

Each row represents a transaction (or daily aggregation) with at least:

* `date`
* `product_name`
* `specification`
* `selling_price`
* `units_sold`
* `revenue`
* `profit`
* `discount`
* Optional flags (season, covid, drought, etc.)

---

## Step-by-Step Pipeline

### 1️⃣ Product Identification

A **unique product_id** is created by concatenating:

```
product_id = product_name + '_' + specification
```

This ensures stable product tracking across time.

---

### 2️⃣ Weekly Aggregation

Data is converted into **week × product** format:

| Column        | Aggregation Logic |
| ------------- | ----------------- |
| selling_price | Mean              |
| units_sold    | Sum               |
| revenue       | Sum               |
| profit        | Sum               |
| discount      | Sum               |

Each `week_id` is unique and time-ordered.

This structure is used for **elasticity modeling** and **optimization**.

---

### 3️⃣ Elasticity Modeling

#### Approach

* Only **own-price elasticities** are used
* Separate elasticity per product
* External factors added as regressors (when available)

Example model:

```
log(units_sold) = β0 + β1 * log(price) + β2 * season + β3 * covid + ε
```

Where:

* `β1` = **price elasticity of demand**

#### Output

A clean elasticity table:

| product_id | elasticity | model_used | r_squared |

---

### 4️⃣ Optimization Data Preparation

From historical data, the **last N weeks** (typically 8–10) are selected to compute:

* Base price (average selling price)
* Base units sold
* Base revenue
* Base profit

This becomes the **baseline** for optimization.

---

### 5️⃣ Business Constraints (config-driven)

All constraints live in `config.py`.

Example:

```python
PRICE_CONSTRAINTS = {
    "default": {
        "max_price_change": 10,        # absolute ₹
        "max_pct_change": 0.10         # 10%
    },
    "premium": {
        "max_price_change": 25,
        "max_pct_change": 0.15
    }
}
```

The **effective max price change** is:

```
min(max_price_change, base_price * max_pct_change)
```

---

### 6️⃣ Revenue Optimization Logic

For each product:

1. Generate candidate prices within allowed bounds
2. Predict demand using elasticity:

```
Q_new = Q_base * (P_new / P_base) ^ elasticity
```

3. Compute:

* New revenue
* Revenue delta
* Profit delta

4. Select price that **maximizes revenue**

---

### 7️⃣ Price Rounding

Optimized prices are rounded using predefined rules (e.g.):

* Nearest integer
* Nearest 5 or 10

Rounding happens **after optimization** to avoid biasing the solution space.

---

## 📊 Analysis Reports

- [Exploratory Data Analysis](reports/eda/EDA.md)
- [Elasticity Analysis](reports/elasticity/Elasticity_Report.md)
- [Price Optimization Results](reports/optimization/Optimization_Report.md)

---

## Final Output

The final optimized dataframe includes:

| Column            | Description              |
| ----------------- | ------------------------ |
| product_id        | Unique product           |
| base_price        | Historical average price |
| optimized_price   | Recommended price        |
| price_change      | Absolute change          |
| base_units        | Historical units         |
| predicted_units   | Expected units           |
| base_revenue      | Historical revenue       |
| optimized_revenue | Expected revenue         |
| revenue_change    | Revenue delta            |
| profit_change     | Profit delta             |

---

## How to Run

1. Install dependencies

```
pip install -r requirements.txt
```

2. Prepare elasticity

* Run elasticity notebooks or scripts

3. Prepare optimization data

4. Run revenue optimizer

5. Review optimized output

---

## Design Decisions & Rationale

* ❌ No cross-elasticity (limited product overlap)
* ✅ Config-driven constraints (business-friendly)
* ✅ Modular pipeline (easy to test & extend)
* ✅ Elasticity first, optimization second (clean separation)

---

## Possible Extensions

* Profit maximization instead of revenue
* Category-level elasticity pooling
* Cross-price elasticity
* Scenario simulations
* Store-level optimization
* Automated model selection

---

## Author Notes

This project was built with a **practical pricing mindset** — focusing on:

* Explainability
* Business realism
* Clean, reusable code

It is designed to be **production-ready** with minimal refactoring.

---

## Contact

For questions, improvements, or discussions around pricing science and optimization, feel free to reach out.

Happy optimizing 🚀
