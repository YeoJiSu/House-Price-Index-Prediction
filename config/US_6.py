import os

"""
Setting for United States
"""
# File path settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Absolute Path
FEATURE_PATH = os.path.join(BASE_DIR, "..", "data", "US_explainable_variables.csv")
TARGET_PATH  = os.path.join(BASE_DIR, "..", "data", "US_target_variables.csv")

# Date settings
TRAIN_DATE     = '2000-01-01'
TEST_DATE      = '2020-01-01'
REFERENCE_DATE = '2000-01-01'   # Base date

# Feature (explanatory variables)
INTEREST_RATE      = ["call_rate", "bond_10yr"] # "federal_dept"
STOCK              = ["NASDAQ"]
MONEY_SUPPLY       = ["M1_velocity", "M2_real", "M2_velocity"]
LIQUIDITY          = ["financial", "nonfinancial"]
INFLATION          = ["CPI", "inflation"]
EMPLOYEE           = ["employee"]
GDP                = ["GDP"]
CONSTRUCTION       = ["total_construction", "IPI_supply", "house_supply"]
FEATURE_COLUMN = INTEREST_RATE + STOCK + MONEY_SUPPLY + LIQUIDITY + INFLATION + EMPLOYEE + GDP + CONSTRUCTION

# Kalman Filter Covariance Setting
KALMAN_OBSERVATION_COV = 10      # Measurement noise
KALMAN_TRANSITION_COV  = 0.01      # State transition noise

# VAR model parameters
MAXLAG_FACTOR    = 23
SELECTED_LAG_FACTOR = 3
FORECAST_HORIZON  = 6  # Forecast steps (e.g., 6-month prediction)
WINDOW_SIZE = 3

# Deep learning model settings
MODEL_NAME     = "PatchLatentMLPV2"
FORECAST_SIZE  = FORECAST_HORIZON 
VERSION        = f"{FORECAST_SIZE}{MODEL_NAME}_{KALMAN_OBSERVATION_COV}_{KALMAN_TRANSITION_COV}"
DIR_PATH       = "/Users/yeojisu/Documents/HPI-Save/exp/US/"

# Common text (if needed)
COMMON_TXT = f"maxlag={MAXLAG_FACTOR}selected_lag={SELECTED_LAG_FACTOR}kalman={KALMAN_OBSERVATION_COV}&{KALMAN_TRANSITION_COV}"
