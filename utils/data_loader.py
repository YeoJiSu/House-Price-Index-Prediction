import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from torch.utils.data import Dataset
import torch

def load_target_data(target_path):
    target_df = pd.read_csv(target_path).T
    target_df.columns = target_df.iloc[0]
    target_df = target_df.iloc[1:]
    # Convert to numeric type and remove missing values
    for col in target_df.columns:
        target_df[col] = pd.to_numeric(target_df[col], errors='coerce')
    target_df = target_df.dropna()
    target_df.index = pd.to_datetime(target_df.index)
    return target_df

def load_target_data_us(target_path):
    target_df = pd.read_csv(target_path).set_index("observation_date")
    for col in target_df.columns:
        target_df[col] = pd.to_numeric(target_df[col], errors='coerce')
    target_df = target_df.dropna()
    target_df.index = pd.to_datetime(target_df.index)
    return target_df

def load_feature_data(feature_path, exp_columns, train_date):
    df = pd.read_csv(feature_path).T
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df[exp_columns].dropna()
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= train_date]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

def load_feature_data_us(feature_path, exp_columns, train_date):
    df = pd.read_csv(feature_path).set_index("observation_date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[exp_columns].dropna()
    df.index = pd.to_datetime(df.index)
    return df

def adf_test(series, name=''):
    if series.isnull().any() or np.isinf(series).any():
        print(f"Error: {name} contains NaN or infinite values.")
        return False
    r = adfuller(series, autolag='AIC')
    p_value = r[1]
    print(f'ADF Statistic for {name}: {r[0]:.4f}')
    print(f'p-value for {name}: {p_value:.4f}')
    return p_value < 0.05

def process_series(series, name):
    print(f"\nProcessing {name}")
    if adf_test(series, name):
        print(f"{name} is already stationary.")
        return series
    # First-order differencing
    diff = series.diff().dropna()
    if diff.empty or not adf_test(diff, f'{name}_1st_diff'):
        # Second-order differencing
        diff2 = diff.diff().dropna()
        if diff2.empty or not adf_test(diff2, f'{name}_2nd_diff'):
            print(f"{name} is still non-stationary after transformations.")
            return None
        else:
            print(f"{name}_2nd_diff is now stationary.")
            return diff2
    else:
        print(f"{name}_1st_diff is now stationary.")
        return diff

def granger_test(data, target, maxlag=12):
    granger_values = {}
    min_p_values = {}
    for col in data.columns:
        if col != target:
            test_result = grangercausalitytests(data[[target, col]], maxlag=maxlag, verbose=False)
            min_p_value = float('inf')
            min_p_lag = 0
            for lag, result in test_result.items():
                p_value = result[0]['ssr_chi2test'][1]
                if p_value < min_p_value:
                    min_p_value = p_value
                    min_p_lag = lag
            if min_p_value <= 0.05:
                min_p_values[col] = min_p_lag
    granger_values[target] = min_p_values
    print(min_p_values)
    return min_p_values

def granger_test_all(data, target, maxlag=12):
    granger_values = {}
    significant_results = {}
    for col in data.columns:
        if col != target:
            test_result = grangercausalitytests(data[[target, col]], maxlag=maxlag, verbose=False)
            sig_lags = {}
            for lag, result in test_result.items():
                p_value = result[0]['ssr_chi2test'][1]
                if p_value <= 0.05:
                    sig_lags[lag] = p_value
            if sig_lags:
                significant_results[col] = sig_lags
    granger_values[target] = significant_results
    return significant_results

def standardization(train_df, test_df, not_col, target_cols):
    train_df_ = train_df.copy()
    test_df_ = test_df.copy()
    cols = [col for col in train_df.columns if col != not_col]
    mean_dict = {}
    std_dict = {}
    for x in cols:
        mean, std = train_df_.agg(["mean", "std"]).loc[:, x]
        train_df_[x] = (train_df_[x] - mean) / std
        test_df_[x] = (test_df_[x] - mean) / std
        if x in target_cols:
            mean_dict[x] = mean
            std_dict[x] = std
    return train_df_, test_df_, mean_dict, std_dict

def z_transform(X, mu, std):
    """
    ==========================
    | X: df, mu: df, std: df |
    ==========================
    """
    return (X-mu)/(std+0.000001)

def inv_z_transform(y, mu, std):
    """
    ====================================
    | y: ndarray, mu: df, std: df      |
    ====================================
    """
    return y * (std+0.000001) + mu

def time_slide_df(df, window_size, forecast_size, date, target_cols):
    data_list, dap_list, date_list = [], [], []
    for idx in range(0, df.shape[0] - window_size - forecast_size + 1):
        # Past window_size period data
        x = df.loc[idx:idx + window_size - 1, target_cols].values
        # Future forecast_size period actual values
        y = df.loc[idx + window_size:idx + window_size + forecast_size - 1, target_cols].values
        # Future dates
        date_ = df.loc[idx + window_size:idx + window_size + forecast_size - 1, date].values
        data_list.append(x)
        dap_list.append(y)
        date_list.append(date_)
    return np.array(data_list, dtype='float32'), np.array(dap_list, dtype='float32'), np.array(date_list)

def test_time_slide_df(df, window_size, forecast_size, date, target_cols):
    data_list, dap_list, date_list = [], [], []
    for idx in range(df.shape[0] - window_size - forecast_size, df.shape[0] - window_size):
        x = df.loc[idx:idx + window_size - 1, target_cols].values
        # Use dummy values
        y = np.array([0] * forecast_size)
        date_ = df.loc[idx:idx + window_size - 1, target_cols].values
        data_list.append(x)
        dap_list.append(y)
        date_list.append(date_)
    return np.array(data_list, dtype='float32'), np.array(dap_list, dtype='float32'), np.array(date_list)

class Data(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y
