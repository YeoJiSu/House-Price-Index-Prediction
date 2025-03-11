import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_var_forecast(df, forecast_df, target, test_date, forecast_horizon):
    plt.figure(figsize=(8, 4))
    plt.plot(pd.to_datetime(df.index), df[target], label="Original", color="blue")
    plt.plot(pd.to_datetime(forecast_df.index), forecast_df[target],
             label=f"Predicted (t+{forecast_horizon})", linestyle="dashed", color="red")
    plt.axvline(pd.to_datetime(test_date), color="black", linestyle="--", label="test start")
    plt.legend()
    plt.title(f"t+{forecast_horizon} {target} Prediction with VAR Model")
    plt.show()

def plot_kalman_filter_result(original_df, smoothed_series, target_name):
    plt.figure(figsize=(8, 4))
    plt.plot(pd.to_datetime(original_df.index), original_df, label="Original", linestyle="dashed", color="red")
    plt.plot(pd.to_datetime(smoothed_series.index), smoothed_series, label="Kalman Smoothed", linewidth=2, color="green")
    plt.legend()
    plt.title(f"Kalman Filter Smoothing on {target_name}")
    plt.show()

def plot_loss_curve(train_loss_list, test_loss_list):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_loss_list) + 1)
    ax1.plot(epochs, train_loss_list, label='Train Loss', color='b')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel("Train Loss", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(epochs, test_loss_list, label='Test Loss', color='r')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel("Test Loss", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc="upper right")
    plt.title('Train and Test Loss Over Epochs')
    fig.tight_layout()
    plt.show()

def plot_forecast_vs_actual(target_df, actual, predicted, dates, f_adjustment, target_name, test_date):
    plt.figure(figsize=(20, 6))
    plt.plot(pd.to_datetime(target_df.index), target_df[target_name], label="Original")
    plt.plot(dates, actual + f_adjustment, label="Actual", color='blue')
    plt.plot(dates, predicted + f_adjustment, label="Predicted", linestyle='--', color='red')
    plt.axvline(pd.to_datetime(test_date), color='black', linestyle=':', linewidth=3)
    plt.title(f'Forecast vs Actuals for {target_name}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
def plot_deep_learning_results(resid, columns_to_use, train_dates, test_dates,
                               train_actual, train_predicted, test_actual, test_predicted,
                               mean_dict, std_dict):
    num_cols = len(columns_to_use)
    plt.figure(figsize=(20, num_cols * 6))
    
    for i, col in enumerate(columns_to_use):
        idv_mean = mean_dict[col]
        idv_std = std_dict[col]
        
        train_actual_rescaled = train_actual[:, i] * idv_std + idv_mean
        train_pred_rescaled   = train_predicted[:, i] * idv_std + idv_mean
        test_actual_rescaled  = test_actual[:, i] * idv_std + idv_mean
        test_pred_rescaled    = test_predicted[:, i] * idv_std + idv_mean
        
        mask = ~pd.isna(test_actual_rescaled) & ~pd.isna(test_pred_rescaled)
        rmse = round(mean_squared_error(test_actual_rescaled[mask], test_pred_rescaled[mask])**0.5, 3)
        mae  = round(mean_absolute_error(test_actual_rescaled[mask], test_pred_rescaled[mask]), 3)
        print(f"{col}: RMSE={rmse}, MAE={mae}")
        
        plt.subplot(num_cols, 1, i + 1)
        plt.plot(pd.to_datetime(resid["Date"]), resid[col], label="Residual", color="gray", alpha=0.7)
        plt.plot(pd.to_datetime(train_dates), train_actual_rescaled, label="Train Actual", color="blue")
        plt.plot(pd.to_datetime(train_dates), train_pred_rescaled, label="Train Predicted", linestyle="--", color="red")
        plt.plot(pd.to_datetime(test_dates), test_actual_rescaled, label="Test Actual", color="blue")
        plt.plot(pd.to_datetime(test_dates), test_pred_rescaled, label="Test Predicted", linestyle="--", color="red")
        plt.axvline(x=pd.to_datetime(test_dates[0]), color="black", linestyle=":", linewidth=3, label="Test Start")
        plt.title(f'Forecast vs Actuals for {col}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        
    plt.tight_layout()
    plt.show()

def plot_additional_test_predictions(target_df, columns_to_use,
                                     train_actual, train_predicted,
                                     test_actual, test_predicted,
                                     addi_test_predicted,
                                     train_dates, test_dates,
                                     mean_dict, std_dict, f_pd_all, forecast_horizon):
    num_cols = len(columns_to_use)
    plt.figure(figsize=(20, num_cols * 6))
    
    for i, col in enumerate(columns_to_use):
        idv_mean = mean_dict[col]
        idv_std = std_dict[col]
        
        # 학습 및 테스트 실제/예측값 복원
        act = np.concatenate((train_actual[:, i], test_actual[:, i])) * idv_std + idv_mean
        pre = np.concatenate((train_predicted[:, i], test_predicted[:, i], addi_test_predicted[:, i])) * idv_std + idv_mean
        dat = np.concatenate((train_dates, test_dates))
        
        last_date = np.concatenate((train_dates,test_dates))[-1]
        new_dat = np.array(pd.date_range(start=last_date, periods=forecast_horizon, freq="MS") + pd.DateOffset(days=14))
        f_pd = pd.DataFrame(f_pd_all.iloc[:,i].values,index=f_pd_all.iloc[:,i].index)
        act_pd = pd.DataFrame(act, index = pd.to_datetime(dat))
        pre_pd = pd.DataFrame(pre, index = pd.to_datetime(np.concatenate((dat,new_dat))))
        
        plt.subplot(num_cols, 1, i + 1)
        
        plt.plot(pd.to_datetime(target_df.index),target_df[target_df.columns[i]])
        plt.plot(act_pd.add(f_pd).index, act_pd.add(f_pd),c='b',label="Actual")
        plt.plot(pre_pd.add(f_pd).index, pre_pd.add(f_pd),c='r',label="Predicted", linestyle='--')
        plt.axvline(x=test_dates[0], color='black', linestyle=':', linewidth=3)
        
        plt.title(f'Forecast vs Actuals for {columns_to_use[i]}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        
    plt.tight_layout()
    plt.show()