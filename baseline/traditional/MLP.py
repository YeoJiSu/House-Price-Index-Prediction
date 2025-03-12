import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len      # Input Sequence
        self.pred_len = configs.pred_len    # Output Sequence
        self.c_in = configs.enc_in          # Number of Channels

        self.linear = nn.Linear(self.seq_len, self.pred_len) # 1-layer MLP
    
    def forward(self, x):
        """
        x: [Batch, Input, Channel]
        output: [Batch, Output, Channel]
        """

        # (B, Input, Channel) → (B, Channel, Input)
        x = x.permute(0, 2, 1)  

        # 1-layer MLP
        x = self.linear(x)  # (B, Channel, Output)

        # 원래 형태로 되돌리기 (B, Output, Channel)
        x = x.permute(0, 2, 1)
        return x
