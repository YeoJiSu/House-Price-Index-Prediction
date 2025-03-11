import pandas as pd
from statsmodels.tsa.api import VAR
from utils.data_loader import granger_test

def var_forecast(df, target, test_date, maxlag, selected_lag, forecast_horizon):
    # First-order difference
    data_diff = df.diff().dropna()
    # Select explanatory variables using Granger test (excluding target)
    granger_values = granger_test(data_diff[data_diff.index < test_date], target, maxlag=maxlag)
    my_cols = list(granger_values.keys()) + [target]
    data_diff = df[my_cols].diff().dropna()
    
    # Training data (before test_date)
    train = data_diff[:test_date]
    var_model = VAR(train)
    var_model_fitted = var_model.fit(selected_lag)
    
    forecasted_levels = []
    forecast_dates = []
    for i in range(len(data_diff) - selected_lag + 1):
        window_data = data_diff.iloc[i : i + selected_lag].values
        forecast_diff = var_model_fitted.forecast(window_data, steps=forecast_horizon)
        forecast_diff_cumsum = forecast_diff.cumsum(axis=0)[-1]
        if i + selected_lag < len(df):
            last_level = df[my_cols].iloc[i + selected_lag]
        else:
            continue
        pred_level = last_level + forecast_diff_cumsum
        forecasted_levels.append(pred_level)
        if i + selected_lag + forecast_horizon < len(df):
            forecast_date = df.index[i + selected_lag + forecast_horizon]
        else:
            last_date = pd.to_datetime(df.index[-1])
            offset_periods = (i + selected_lag + forecast_horizon) - (len(df) - 1)
            forecast_date = last_date + pd.DateOffset(months=offset_periods)
        forecast_dates.append(forecast_date)
    forecast_df = pd.DataFrame(forecasted_levels, index=forecast_dates, columns=df[my_cols].columns)
    return forecast_df
