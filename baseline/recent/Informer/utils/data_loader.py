import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
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

from datetime import timedelta
# 시간 특징을 freq에 따라 추출
def time_features(dates, freq='m'):
    dates['year'] = dates.date.apply(lambda x: x.year)
    dates['month'] = dates.date.apply(lambda row:row.month,1)
    dates['day'] = dates.date.apply(lambda row:row.day,1)
    dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
    dates['hour'] = dates.date.apply(lambda row:row.hour,1)
    dates['minute'] = dates.date.apply(lambda row:row.minute,1)
    dates['minute'] = dates.minute.map(lambda x:x//15)
    freq_map = {
        'y':[],'m': ['year', 'month'],'w':['month'],'d':['month','day','weekday'],
        'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
        't':['month','day','weekday','hour','minute'],
    }
    return dates[freq_map[freq.lower()]].values
class Dataset_Pred(Dataset):
    def __init__(self, dataframe, size=None, scale=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dataframe = dataframe
        
        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self.dataframe
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        delta = df_raw["date"].iloc[1] - df_raw["date"].iloc[0]
        if delta >= timedelta(days=28):  # 데이터가 월별로 제공된다고 가정
            self.freq = 'm'
        else:
            self.freq = 'd'  # 기본값을 일별로 설정
        border1 = 0
        border2 = len(df_raw)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]


        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1