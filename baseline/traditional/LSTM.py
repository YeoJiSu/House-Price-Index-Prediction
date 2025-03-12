import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len    # Input Sequence
        self.pred_len = configs.pred_len  # Output Sequence
        self.c_in = configs.enc_in        # Number of Channels

        # LSTM
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, self.pred_len)

    def forward(self, x):
        """
        x: [Batch, seq_len, Channel]
        output: [Batch, pred_len, Channel]
        """
        # (B, seq_len, Channel) → (B, Channel, seq_len)
        x = x.permute(0, 2, 1)
        B, C, T = x.shape  # T == seq_len

        # (B, Channel, seq_len) → (B * Channel, seq_len, 1)
        x = x.reshape(B * C, T, 1)
        
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers, B * Channel, hidden_size]
        h_last = h_n[-1]           # [B * Channel, hidden_size]
        
        # (B * Channel, pred_len)
        pred = self.fc(h_last)
        # (B * Channel, pred_len) → (B, Channel, pred_len)
        pred = pred.reshape(B, C, self.pred_len)
        # (B, Channel, pred_len) → (B, pred_len, Channel)
        pred = pred.permute(0, 2, 1)
        return pred
