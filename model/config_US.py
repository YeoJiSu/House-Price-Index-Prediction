import os

"""
Setting for United States
"""
# File path settings
FEATURE_PATH = os.path.join("..", "data", "US_explainable_variables.csv")
TARGET_PATH  = os.path.join("..", "data", "US_target_variables.csv")

# Date settings
TRAIN_DATE     = '2000-01-01'
TEST_DATE      = '2020-01-01'
REFERENCE_DATE = '2000-01-01'   # Base date

# Kalman Filter Covariance Setting
KALMAN_OBSERVATION_COV = 1      # Measurement noise
KALMAN_TRANSITION_COV  = 1      # State transition noise

# VAR model parameters
MAXLAG_FACTOR    = 23
SELECTED_LAG_FACTOR = 3
FORECAST_HORIZON  = 3  # Forecast steps (e.g., 6-month prediction)
WINDOW_SIZE = 3

# Deep learning model settings
MODEL_NAME     = "PatchLatentMLP"
FORECAST_SIZE  = FORECAST_HORIZON 
VERSION        = f"{FORECAST_SIZE}{MODEL_NAME}_{KALMAN_OBSERVATION_COV}_{KALMAN_TRANSITION_COV}"
DIR_PATH       = "/Users/yeojisu/Documents/HPI-Save/example/"

# Common text (if needed)
COMMON_TXT = f"maxlag={MAXLAG_FACTOR}selected_lag={SELECTED_LAG_FACTOR}kalman={KALMAN_OBSERVATION_COV}&{KALMAN_TRANSITION_COV}"
