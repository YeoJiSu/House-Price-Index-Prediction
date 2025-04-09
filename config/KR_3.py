import os

"""
Setting for South Korea
"""
# File path settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Absolute Path
FEATURE_PATH = os.path.join(BASE_DIR, "..", "data", "KR_explainable_variables.csv")
TARGET_PATH  = os.path.join(BASE_DIR, "..", "data", "KR_target_variables.csv")

# Date settings
TRAIN_DATE     = '2006-01-15'
TEST_DATE      = '2021-01-15'
REFERENCE_DATE = '2017-11-15'   # Base date

# Feature (explanatory variables)
INTEREST_RATE      = ["call_rate", "bond_3yr", "loan_rate_avg"]
STOCK              = ["KOSPI", "KOSDAQ", "NASDAQ"]
MONEY_SUPPLY       = ["M2_KR", "M2_US"]
LIQUIDITY          = ["broad_liquidity"]
INFLATION          = ["CPI_growth"]
EXCHANGE_RATE      = ["USD_KRW_rate"]
GDP                = ["GDP_growth"]
CONSTRUCTION       = ["res_start", "res_permit"]
FEATURE_COLUMN = INTEREST_RATE + STOCK + MONEY_SUPPLY + LIQUIDITY + INFLATION + EXCHANGE_RATE + GDP + CONSTRUCTION

# Kalman Filter Covariance Setting
KALMAN_OBSERVATION_COV = 10      # Measurement noise
KALMAN_TRANSITION_COV  = 0.01    # State transition noise

# VAR model parameters
MAXLAG_FACTOR    = 23
SELECTED_LAG_FACTOR = 3
FORECAST_HORIZON  = 3  # Forecast steps (e.g., 6-month prediction)
WINDOW_SIZE = 3

# Deep learning model settings
MODEL_NAME     = "PatchLatentMLPV3"
FORECAST_SIZE  = FORECAST_HORIZON 
VERSION        = f"{FORECAST_SIZE}{MODEL_NAME}_{KALMAN_OBSERVATION_COV}_{KALMAN_TRANSITION_COV}"
DIR_PATH       = "/Users/yeojisu/Documents/HPI-Save/exp/KR/"

# Common text (if needed)
COMMON_TXT = f"maxlag={MAXLAG_FACTOR}selected_lag={SELECTED_LAG_FACTOR}kalman={KALMAN_OBSERVATION_COV}&{KALMAN_TRANSITION_COV}"
