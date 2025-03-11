import pandas as pd
from pykalman import KalmanFilter

def apply_kalman_filter(series, observation_covariance, transition_covariance):
    kf = KalmanFilter(
        initial_state_mean=series.iloc[0],
        n_dim_obs=1,
        transition_matrices=[1],
        observation_matrices=[1],
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    filtered_state_means, _ = kf.filter(series.values)
    smoothed_series = pd.Series(filtered_state_means.flatten(), index=series.index)
    return smoothed_series
