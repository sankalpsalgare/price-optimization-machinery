import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# =========================
# DATE CONFIG
# =========================

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
BASE_YEAR = 2019
DATES = pd.date_range(START_DATE, END_DATE, freq="D")

# =========================
# PRODUCT MASTER (EXPANDED)
# =========================

PRODUCTS = [
    # PVC Pipes – Supreme style
    {"brand": "Supreme", "category": "Pipe", "product": "PVC Pipe", "spec": "1 inch", "base_price": 120},
    {"brand": "Supreme", "category": "Pipe", "product": "PVC Pipe", "spec": "1.5 inch", "base_price": 170},
    {"brand": "Supreme", "category": "Pipe", "product": "PVC Pipe", "spec": "2 inch", "base_price": 220},
    {"brand": "Supreme", "category": "Pipe", "product": "PVC Pipe", "spec": "3 inch", "base_price": 380},
    {"brand": "Supreme", "category": "Pipe", "product": "PVC Pipe", "spec": "4 inch", "base_price": 620},

    # Fittings
    {"brand": "Supreme", "category": "Fitting", "product": "PVC Elbow", "spec": "Standard", "base_price": 35},
    {"brand": "Supreme", "category": "Fitting", "product": "PVC Tee", "spec": "Standard", "base_price": 45},
    {"brand": "Supreme", "category": "Fitting", "product": "PVC Coupler", "spec": "Standard", "base_price": 30},

    # Irrigation
    {"brand": "Jain", "category": "Irrigation", "product": "Drip Kit", "spec": "1 Acre", "base_price": 1500},
    {"brand": "Jain", "category": "Irrigation", "product": "Sprinkler Set", "spec": "Heavy Duty", "base_price": 2200},

    # Motors – Shakti style
    {"brand": "Shakti", "category": "Motor", "product": "Motor Pump", "spec": "1 HP Single Phase", "base_price": 9500},
    {"brand": "Shakti", "category": "Motor", "product": "Motor Pump", "spec": "2 HP Single Phase", "base_price": 14500},
    {"brand": "Shakti", "category": "Motor", "product": "Motor Pump", "spec": "3 HP Three Phase", "base_price": 22000},
]

# =========================
# CUSTOMER TYPES
# =========================

CUSTOMERS = {
    "Retail": 1.0,
    "Farmer": 1.4,
    "Contractor": 1.8,
    "Government": 2.2
}

# =========================
# SEASONALITY (INDIA-CORRECT)
# =========================

def get_season(date):
    if date.month in [2, 3, 4, 5]:
        return "Pre-Monsoon"
    elif date.month in [6, 7, 8, 9]:
        return "Monsoon"
    elif date.month in [10, 11]:
        return "Post-Monsoon"
    else:
        return "Winter"

SEASON_FACTOR = {
    "Pre-Monsoon": 1.6,
    "Monsoon": 1.2,
    "Post-Monsoon": 1.0,
    "Winter": 0.75
}

# =========================
# INFLATION
# =========================

INFLATION = {
    "Pipe": 0.07,
    "Fitting": 0.05,
    "Irrigation": 0.06,
    "Motor": 0.08
}

def inflate_price(price, category, date):
    years = date.year - BASE_YEAR
    inflation = (1 + INFLATION[category]) ** years
    noise = np.random.normal(1.0, 0.03)
    return price * inflation * noise

# =========================
# EXTERNAL PRICE SHOCKS
# =========================

def generate_shocks(start, end):
    shocks = []
    cur = start
    while cur < end:
        duration = random.randint(120, 180)
        multiplier = np.random.normal(1.0, 0.08)
        shocks.append((cur, cur + timedelta(days=duration), multiplier))
        cur += timedelta(days=duration)
    return shocks

SHOCKS = generate_shocks(
    datetime.strptime(START_DATE, "%Y-%m-%d"),
    datetime.strptime(END_DATE, "%Y-%m-%d")
)

def apply_shock(price, date):
    for s, e, m in SHOCKS:
        if s <= date <= e:
            return price * m
    return price

# =========================
# DATA GENERATION
# =========================

records = []
invoice_id = 100000

for date in DATES:
    season = get_season(date)

    for prod in PRODUCTS:
        if prod["category"] in ["Pipe", "Motor", "Irrigation"]:
            base_demand = random.randint(3, 7) * SEASON_FACTOR[season]
        else:
            base_demand = random.randint(1, 4)

        for cust, mult in CUSTOMERS.items():
            qty = int(np.random.poisson(base_demand * mult))
            if qty <= 0:
                continue

            invoice_id += 1

            base_price = inflate_price(prod["base_price"], prod["category"], date)
            base_price = apply_shock(base_price, date)

            discount = np.random.uniform(0.02, 0.15)
            selling_price = round(base_price * (1 - discount), 2)
            cost = round(selling_price * np.random.uniform(0.7, 0.85), 2)

            records.append({
                "invoice_id": invoice_id,
                "invoice_date": date,
                "year": date.year,
                "month": date.month,
                "season": season,
                "customer_type": cust,
                "brand": prod["brand"],
                "product_name": prod["product"],
                "product_category": prod["category"],
                "specification": prod["spec"],
                "quantity": qty,
                "mrp": round(base_price, 2),
                "selling_price": selling_price,
                "unit_cost": cost,
                "discount_pct": round(discount * 100, 2),
                "revenue": round(qty * selling_price, 2),
                "profit": round(qty * (selling_price - cost), 2)
            })

# =========================
# FINAL DATAFRAME
# =========================

df = pd.DataFrame(records)
df.sort_values("invoice_date", inplace=True)
df.to_csv("./data/indian_machinery_transaction_data.csv", index=False)

print("✅ Dataset generated")
print("Rows:", len(df))
print("From:", df.invoice_date.min(), "To:", df.invoice_date.max())
