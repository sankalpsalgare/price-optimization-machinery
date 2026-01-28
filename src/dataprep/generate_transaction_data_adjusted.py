import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ------------------
# Date Range
# ------------------
START_DATE = "2019-01-01"
END_DATE = "2025-06-30"
DATES = pd.date_range(START_DATE, END_DATE, freq="D")

# ------------------
# Product Master
# ------------------
# PRODUCTS = [
#     {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"1 inch","base_price":120},
#     {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"1.5 inch","base_price":170},
#     {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"2 inch","base_price":220},
#     {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"3 inch","base_price":380},
#     {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"4 inch","base_price":620},
#     {"brand":"Supreme","category":"Fitting","product":"PVC Elbow","spec":"Standard","base_price":35},
#     {"brand":"Supreme","category":"Fitting","product":"PVC Tee","spec":"Standard","base_price":45},
#     {"brand":"Supreme","category":"Fitting","product":"PVC Coupler","spec":"Standard","base_price":30},
#     {"brand":"Jain","category":"Irrigation","product":"Drip Kit","spec":"1 Acre","base_price":1500},
#     {"brand":"Jain","category":"Irrigation","product":"Sprinkler Set","spec":"Heavy Duty","base_price":2200},
#     {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"1 HP Single Phase","base_price":9500},
#     {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"2 HP Single Phase","base_price":14500},
#     {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"3 HP Three Phase","base_price":22000}
# ]
PRODUCTS = [

    # =====================
    # PVC PIPES – SUPREME
    # =====================
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"0.5 inch","base_price":90},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"0.75 inch","base_price":105},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"1 inch","base_price":120},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"1.25 inch","base_price":145},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"1.5 inch","base_price":170},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"2 inch","base_price":220},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"2.5 inch","base_price":300},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"3 inch","base_price":380},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"4 inch","base_price":620},
    {"brand":"Supreme","category":"Pipe","product":"PVC Pipe","spec":"6 inch","base_price":980},

    # =====================
    # CPVC / AGRI PIPES
    # =====================
    {"brand":"Supreme","category":"Pipe","product":"CPVC Pipe","spec":"1 inch","base_price":180},
    {"brand":"Supreme","category":"Pipe","product":"CPVC Pipe","spec":"1.5 inch","base_price":260},
    {"brand":"Supreme","category":"Pipe","product":"CPVC Pipe","spec":"2 inch","base_price":360},

    # =====================
    # FITTINGS
    # =====================
    {"brand":"Supreme","category":"Fitting","product":"PVC Elbow","spec":"0.5 inch","base_price":25},
    {"brand":"Supreme","category":"Fitting","product":"PVC Elbow","spec":"1 inch","base_price":35},
    {"brand":"Supreme","category":"Fitting","product":"PVC Tee","spec":"1 inch","base_price":45},
    {"brand":"Supreme","category":"Fitting","product":"PVC Coupler","spec":"1 inch","base_price":30},
    {"brand":"Supreme","category":"Fitting","product":"PVC End Cap","spec":"1 inch","base_price":20},
    {"brand":"Supreme","category":"Fitting","product":"PVC Union","spec":"1 inch","base_price":65},

    # =====================
    # IRRIGATION SYSTEMS
    # =====================
    {"brand":"Jain","category":"Irrigation","product":"Drip Irrigation Kit","spec":"0.5 Acre","base_price":900},
    {"brand":"Jain","category":"Irrigation","product":"Drip Irrigation Kit","spec":"1 Acre","base_price":1500},
    {"brand":"Jain","category":"Irrigation","product":"Drip Irrigation Kit","spec":"2 Acre","base_price":2600},
    {"brand":"Jain","category":"Irrigation","product":"Sprinkler Set","spec":"Mini","base_price":1400},
    {"brand":"Jain","category":"Irrigation","product":"Sprinkler Set","spec":"Heavy Duty","base_price":2200},

    # =====================
    # MOTOR PUMPS – SHAKTI
    # =====================
    {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"0.5 HP Single Phase","base_price":7200},
    {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"1 HP Single Phase","base_price":9500},
    {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"1.5 HP Single Phase","base_price":11800},
    {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"2 HP Single Phase","base_price":14500},
    {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"3 HP Three Phase","base_price":22000},
    {"brand":"Shakti","category":"Motor","product":"Motor Pump","spec":"5 HP Three Phase","base_price":36500},

    # =====================
    # ACCESSORIES (LOW VALUE, HIGH VOLUME)
    # =====================
    {"brand":"Generic","category":"Fitting","product":"Pipe Clamp","spec":"Standard","base_price":18},
    {"brand":"Generic","category":"Fitting","product":"Thread Tape","spec":"Standard","base_price":12},
    {"brand":"Generic","category":"Fitting","product":"Foot Valve","spec":"1 inch","base_price":320},
]

# ------------------
# Customer Types with Imbalance
# ------------------
CUSTOMERS = {
    "Retail": 0.2,
    "Farmer": 0.62,
    "Contractor": 0.2,
    "Government": 0.08
}

# ------------------
# Seasonality Function
# ------------------
def get_season(date):
    if date.month in [2,3,4,5]:
        return "Pre-Monsoon"
    elif date.month in [6,7,8,9]:
        return "Monsoon"
    elif date.month in [10,11]:
        return "Post-Monsoon"
    else:
        return "Winter"

SEASON_FACTORS = {
    "Pre-Monsoon": 1.8,
    "Monsoon": 1.2,
    "Post-Monsoon": 1.0,
    "Winter": 0.6
}

# ------------------
# Inflation and Deflation Logic
# ------------------
CATEGORY_INFLATION = {
    "Pipe": 0.07,
    "Fitting": 0.05,
    "Irrigation": 0.06,
    "Motor": 0.08
}

def price_with_inflation(base, category, date):
    year = date.year
    years_passed = year - 2019
    inflation = (1 + CATEGORY_INFLATION[category]) ** years_passed
    
    # Simulate 2023→2024 deflation 
    if year == 2024:
        inflation *= 0.97
    
    noise = np.random.normal(1.0, 0.03)
    return base * inflation * noise

# ------------------
# External Shocks Setup
# ------------------
def create_shocks(start, end):
    shock_periods = []
    current = start
    while current < end:
        duration = random.randint(100, 180)
        multiplier = np.random.normal(1.0, 0.08)
        shock_periods.append((current, current + timedelta(days=duration), multiplier))
        current += timedelta(days=duration)
    return shock_periods

SHOCK_PERIODS = create_shocks(
    datetime.strptime(START_DATE,"%Y-%m-%d"),
    datetime.strptime(END_DATE,"%Y-%m-%d")
)

def apply_shock(price, date):
    for s,e,m in SHOCK_PERIODS:
        if s <= date <= e:
            return price * m
    return price

# ------------------
# Realistic Demand Modifiers for Drought Year
# ------------------
def macro_demand_modifier(date):
    """
    Simulates COVID + recovery impact on demand
    """
    if date.year == 2019:
        return 1.0
    elif date.year == 2020:
        return 0.55   # lockdown impact
    elif date.year == 2021:
        return 0.45   # worst year
    elif date.year == 2022:
        return 0.75   # recovery starts (FY22)
    elif date.year == 2023:
        return 0.95
    else:
        return 1.05   # growth phase (2024–2025)

# ------------------
# Data Generation
# ------------------
records = []
invoice_id = 500000

for date in DATES:
    season = get_season(date)
    macro_factor = macro_demand_modifier(date)
    
    for prod in PRODUCTS:
        base_price = price_with_inflation(prod["base_price"], prod["category"], date)
        base_price = apply_shock(base_price, date)
        
        for cust_type, cust_weight in CUSTOMERS.items():
            demand_base = np.random.uniform(5,10) * SEASON_FACTORS[season]
            demand = demand_base * cust_weight * macro_factor
            
            qty = np.random.poisson(demand)
            if qty <= 0:
                continue
            
            invoice_id += 1
            # Selling price with some occasional discount or loss
            discount = np.random.uniform(-0.05, 0.18)   # negative allows loss
            sell_price = round(base_price * (1 - discount), 2)
            cost_price = round(base_price * np.random.uniform(0.70, 0.85), 2)
            
            revenue = round(qty * sell_price,2)
            profit = round(qty * (sell_price - cost_price),2)
            
            records.append({
                "invoice_id": invoice_id,
                "invoice_date": date,
                "year": date.year,
                "month": date.month,
                "season": season,
                "customer_type": cust_type,
                "brand": prod["brand"],
                "product_name": prod["product"],
                "product_category": prod["category"],
                "specification": prod["spec"],
                "quantity": qty,
                "mrp": round(base_price,2),
                "selling_price": sell_price,
                "unit_cost": cost_price,
                "discount_pct": round(discount*100,2),
                "revenue": revenue,
                "profit": profit
            })

# ------------------
# Final Dataframe
# ------------------
df = pd.DataFrame(records)
df.sort_values("invoice_date", inplace=True)
df.to_csv("./data/raw/indian_machinery_transaction_data_3.csv", index=False)

print("✨ REALISTIC DATA GENERATED")
print("Rows:", len(df))
print("Date Range:", df.invoice_date.min(), "to", df.invoice_date.max())
