import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.window_size = configs.seq_len
        self.forecast_size = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

    def forward(self, x):
        """
        x: (B, L, C)
        - B: batch size
        - L: seq_len
        - C: enc_in
        """
        # Extract only last value: (B, 1, C)
        seq_last = x[:, -1:, :]

        # Rpeat last value for forecast_size (B, forecast_size, C)
        x = seq_last.repeat(1, self.forecast_size, 1)

        return x
