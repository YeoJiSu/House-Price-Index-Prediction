import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len      # Input Sequence
        self.pred_len = configs.pred_len    # Output Sequence
        self.c_in = configs.enc_in          # Number of Channels

        self.fc1 = nn.Linear(self.seq_len, 128)  
        self.fc2 = nn.Linear(128, 64) 
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, self.pred_len) 

    def forward(self, x):
        """
        x: [Batch, Input, Channel]
        output: [Batch, Output, Channel]
        """

        # (B, Input, Channel) → (B, Channel, Input)
        x = x.permute(0, 2, 1)  

        # MLP 연산
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # (B, Channel, Output)

        # (B, Output, Channel)
        x = x.permute(0, 2, 1)
        return x
