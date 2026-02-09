COVID_YEARS = [2020, 2021, 2022]
DROUGHT_YEARS = [2018, 2019,2020]

PEAK_SEASON_MONTHS = [5, 6, 7, 8]   # Pre & monsoon
OFF_SEASON_MONTHS = [11, 12, 1]

CUSTOMER_PRIORITY = [
    "Farmer",
    "Retail",
    "Contractor"
]

# config.py

PRICING_CONSTRAINTS = {
    "pipe": {
        "max_abs_increase": 10,      # ₹10
        "max_pct_increase": 0.10,    # 10%
        "max_unit_drop": 0.05        # 5%
    },
    "irrigation": {
        "max_abs_increase": 10,      # ₹10
        "max_pct_increase": 0.10,    # 10%
        "max_unit_drop": 0.05        # 5%
    },
    "fitting": {
        "max_abs_increase": 10,      # ₹10
        "max_pct_increase": 0.10,    # 10%
        "max_unit_drop": 0.05        # 5%
    },
    "motor": {
        "max_abs_increase": 200,     # ₹200
        "max_pct_increase": 0.10,    # 10%
        "max_unit_drop": 0.03        # 5%
    }
}

DEFAULT_ITERATIONS = 150

