import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# =========================================================
# CONFIG
# =========================================================
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"
N_STORES = 25
BASE_YEAR = 2019

INFLATION_RATE = {
    "Pipe": 0.07,
    "Fitting": 0.05,
    "Irrigation": 0.06,
    "Motor": 0.08
}

CUSTOMER_TYPES = ["Retail", "Contractor", "Farmer"]
CUSTOMER_ELASTICITY = {
    "Retail": 1.2,
    "Contractor": 0.8,
    "Farmer": 0.9
}

SEASONALITY = {
    "Pre-Monsoon": 1.6,   # TRUE peak (planning & stocking)
    "Monsoon": 1.2,       # execution & urgent demand
    "Post-Monsoon": 1.0,  # tapering
    "Winter": 0.75        # lowest demand
}

# =========================================================
# PRODUCT MASTER
# =========================================================
PRODUCTS = [
    # -------- Supreme Gold (UPVC) --------
    ("SG_UPVC_15","Supreme Gold","UPVC Pipe SCH-80","Pipe",95,150,1.1),
    ("SG_UPVC_20","Supreme Gold","UPVC Pipe SCH-80","Pipe",140,220,1.1),
    ("SG_UPVC_25","Supreme Gold","UPVC Pipe SCH-80","Pipe",210,330,1.0),
    ("SG_UPVC_32","Supreme Gold","UPVC Pipe SCH-80","Pipe",320,520,0.95),
    ("SG_UPVC_40","Supreme Gold","UPVC Pipe SCH-80","Pipe",480,760,0.9),
    ("SG_UPVC_50","Supreme Gold","UPVC Pipe SCH-80","Pipe",820,1250,0.85),

    # -------- Supreme Gold Fittings --------
    ("SG_ELB_25","Supreme Gold","Elbow 90°","Fitting",12,25,1.6),
    ("SG_TEE_25","Supreme Gold","Tee Joint","Fitting",18,35,1.5),
    ("SG_CPL_32","Supreme Gold","Coupler","Fitting",22,45,1.5),
    ("SG_BND_40","Supreme Gold","Bend 45°","Fitting",35,70,1.4),

    # -------- Shakti Pipes --------
    ("SH_PVC_25","Shakti","R-PVC Pipe","Pipe",130,200,1.2),
    ("SH_PVC_32","Shakti","R-PVC Pipe","Pipe",180,280,1.15),
    ("SH_PVC_40","Shakti","R-PVC Pipe","Pipe",260,420,1.1),
    ("SH_PVC_50","Shakti","R-PVC Pipe","Pipe",420,680,1.05),
    ("SH_PVC_75","Shakti","R-PVC Pipe","Pipe",900,1400,1.0),

    ("SH_CPVC_20","Shakti","CPVC Pipe","Pipe",260,420,0.95),
    ("SH_CPVC_25","Shakti","CPVC Pipe","Pipe",340,560,0.95),
    ("SH_CPVC_32","Shakti","CPVC Pipe","Pipe",520,860,0.9),

    ("SH_SWR_75","Shakti","SWR Pipe Type B","Pipe",620,950,1.0),
    ("SH_SWR_110","Shakti","SWR Pipe Type B","Pipe",980,1500,0.95),

    # -------- Irrigation --------
    ("IR_DRIP_SM","Generic","Drip Irrigation Kit Small","Irrigation",750,1200,0.85),
    ("IR_DRIP_LG","Generic","Drip Irrigation Kit Large","Irrigation",1450,2300,0.8),
    ("IR_SUCTION_1","Generic","Suction Hose","Irrigation",420,700,1.0),
    ("IR_SUCTION_2","Generic","Suction Hose Heavy","Irrigation",650,1050,0.95),

    # -------- Motor Pumps (High Profit) --------
    ("MP_05","Generic","Motor Pump 0.5 HP","Motor",4200,6200,0.6),
    ("MP_10","Generic","Motor Pump 1 HP","Motor",6200,9000,0.55),
    ("MP_15","Generic","Motor Pump 1.5 HP","Motor",8500,12500,0.5),
    ("MP_20","Generic","Motor Pump 2 HP","Motor",11200,16500,0.45),

    # -------- Accessories --------
    ("AC_SOL","Generic","Solvent Cement","Accessory",90,160,1.7),
    ("AC_CLAMP","Generic","Pipe Clamp","Accessory",25,55,1.8),
    ("AC_VALVE","Generic","Ball Valve","Accessory",210,380,1.3),
    ("AC_FOOT","Generic","Foot Valve","Accessory",380,650,1.2),
]

PRODUCT_DF = pd.DataFrame(
    PRODUCTS,
    columns=["product_id","brand","product_name","category","cost","base_price","elasticity"]
)

# =========================================================
# HELPERS
# =========================================================

def apply_inflation(base_price, category, date):
    years_passed = date.year - BASE_YEAR
    inflation = (1 + INFLATION_RATE[category]) ** years_passed
    
    noise = np.random.normal(1.0, 0.03)  # ±3% noise
    return round(base_price * inflation * noise, 2)


def get_season(date):
    if date.month in [2, 3, 4, 5]:
        return "Pre-Monsoon"
    elif date.month in [6, 7, 8, 9]:
        return "Monsoon"
    elif date.month in [10, 11]:
        return "Post-Monsoon"
    else:
        return "Winter"

def date_range(start, end):
    return [start + timedelta(days=i) for i in range((end-start).days + 1)]

# =========================================================
# GENERATION
# =========================================================
records = []

dates = date_range(
    datetime.strptime(START_DATE,"%Y-%m-%d"),
    datetime.strptime(END_DATE,"%Y-%m-%d")
)

for date in dates:
    season = get_season(date)
    for store in range(1, N_STORES+1):
        for _, p in PRODUCT_DF.iterrows():

            # Motor pumps sell less frequently
            base_prob = 0.25 if p["category"] == "Motor" else 0.55
            if np.random.rand() > base_prob:
                continue

            customer = np.random.choice(CUSTOMER_TYPES, p=[0.4,0.35,0.25])
            promo = np.random.rand() < 0.15

            price = p["base_price"] * np.random.normal(1.0, 0.08)
            competitor_price = price * np.random.normal(1.02, 0.05)

            base_demand = np.random.randint(1,4) if p["category"]=="Motor" else np.random.randint(5,20)

            elasticity = p["elasticity"] * CUSTOMER_ELASTICITY[customer]
            promo_factor = 1.25 if promo else 1.0
            # Category-specific seasonal sensitivity
            if p["category"] in ["Irrigation", "Motor", "Pipe"]:
                season_factor = SEASONALITY[season]
            else:
                season_factor = 1.0  # accessories, fittings remain stable
            demand = (
                base_demand *
                np.exp(-elasticity * (price / p["base_price"] - 1)) *
                season_factor *
                promo_factor *
                np.random.normal(1.0,0.1)
            )

            qty = max(0, int(round(demand)))
            if qty == 0:
                continue

            revenue = qty * price
            profit = qty * (price - p["cost"])

            records.append([
                date.date(), store, p["product_id"], p["brand"], p["product_name"],
                p["category"], customer, season, round(p["cost"],2),
                round(p["base_price"],2), round(price,2),
                round(competitor_price,2), promo,
                qty, round(revenue,2), round(profit,2)
            ])

# =========================================================
# SAVE
# =========================================================
df = pd.DataFrame(records, columns=[
    "date","store_id","product_id","brand","product_name","category",
    "customer_type","season","cost","base_price","actual_price",
    "competitor_price","promotion_flag","quantity_sold","revenue","profit"
])

df.to_csv("./data/indian_machinery_pricing_dataset.csv", index=False)

print("DATASET GENERATED")
print("Rows:", len(df))
print(df.head())
