import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len       # Input sequence length
        self.pred_len = configs.pred_len     # Prediction sequence length
        self.c_in = configs.enc_in           # Number of channels
        self.use_patch = configs.use_patch   
        
        # Settings for extracting latent features using Transformer
        self.d_model = getattr(configs, 'd_model', 64)   # Transformer embedding dimension
        self.nhead = getattr(configs, 'nhead', 4)          # Number of heads in multi-head attention
        self.num_layers = getattr(configs, 'num_layers', 3)  # Number of Transformer encoder layers

        self.latent_proj = nn.Linear(1, self.d_model)
        # Transformer Encoder 
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_layers)
        
        if self.use_patch:
            # --- Patch-based processing setup ---
            self.patch_size = configs.patch_size
            # Assumes seq_len is a multiple of patch_size (otherwise needs padding or trimming)
            assert self.seq_len % self.patch_size == 0, "seq_len must be divisible by patch_size."
            self.n_patches = self.seq_len // self.patch_size

            # Convert each patch (length: patch_size) to a 128-dimensional embedding
            self.patch_embedding = nn.Linear(self.patch_size, 128)
            
            # Combined dimension: flatten patch embedding (128 * n_patches) + latent (d_model)
            shared_dim = 128 * self.n_patches + self.d_model
            
            # Shared module for extracting common features before branching
            self.shared_module = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, shared_dim)
            )
            
            # Aggregator MLPs for predicting Trend and Residual separately
            self.trend_aggregator = nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.pred_len)
            )
            self.residual_aggregator = nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.pred_len)
            )
        else:
            # --- Standard MLP-based processing ---
            # Combined dimension: seq_len + d_model
            shared_dim = self.seq_len + self.d_model
            self.shared_module = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, shared_dim)
            )
            
            # Trend
            self.trend_fc1 = nn.Linear(shared_dim, 128)
            self.trend_fc2 = nn.Linear(128, 64)
            self.trend_fc3 = nn.Linear(64, 32)
            self.trend_fc4 = nn.Linear(32, 16)
            self.trend_fc5 = nn.Linear(16, self.pred_len)
            
            # Residual
            self.residual_fc1 = nn.Linear(shared_dim, 128)
            self.residual_fc2 = nn.Linear(128, 64)
            self.residual_fc3 = nn.Linear(64, 32)
            self.residual_fc4 = nn.Linear(32, 16)
            self.residual_fc5 = nn.Linear(16, self.pred_len)
    
    def compute_latent(self, x):
        """
        Extracts latent features using Transformer.
        Args:
            x: Input tensor of shape [B, seq_len, C]
        Returns:
            latent: Tensor of shape [B, C, d_model] - Latent vector per channel.
        """
        B, L, C = x.size()  # x: [B, L, C]
        # Reorder dimensions to process per channel: (B, L, C) → (B, C, L)
        x_perm = x.permute(0, 2, 1)  # [B, C, L]
        # Reshape to merge B and C: [B*C, L] → [B*C, L, 1]
        x_reshaped = x_perm.contiguous().view(B * C, L).unsqueeze(-1)
        # Project scalar values at each time step to d_model dimension
        x_proj = self.latent_proj(x_reshaped)  # [B*C, L, d_model]
        # Convert to Transformer input shape [L, B*C, d_model]
        x_proj = x_proj.permute(1, 0, 2)
        # Pass through Transformer Encoder (captures temporal patterns)
        x_trans = self.transformer_encoder(x_proj)
        # Perform mean pooling over the time dimension to generate latent vectors per channel
        latent = x_trans.mean(dim=0)  # [B*C, d_model]
        # Restore original batch and channel dimensions: [B, C, d_model]
        latent = latent.view(B, C, self.d_model)
        return latent

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, seq_len, C]
        Returns:
            out: Output tensor of shape [B, pred_len, C]
        """
        B, L, C = x.size()
        # Extract latent features using Transformer
        latent = self.compute_latent(x)  # [B, C, d_model]
        
        if self.use_patch:
            # --- Patch-based processing ---
            x_patch = x.permute(0, 2, 1)  # [B, C, seq_len]
            x_patch = x_patch[:, :, :self.n_patches * self.patch_size].view(B, C, self.n_patches, self.patch_size)
            x_patch = self.patch_embedding(x_patch)  # [B, C, n_patches, 128]
            x_patch = torch.relu(x_patch)
            x_patch = x_patch.view(B, C, -1)  # Flatten patches: [B, C, 128 * n_patches]
            x_combined = torch.cat([x_patch, latent], dim=-1)  # Concatenate with latent features
            shared_features = self.shared_module(x_combined)  # [B, C, shared_dim]
            
            trend = self.trend_aggregator(shared_features)  # [B, C, pred_len]
            residual = self.residual_aggregator(shared_features)  # [B, C, pred_len]
            out = trend + residual  # Final prediction
            out = out.permute(0, 2, 1)  # Rearrange to [B, pred_len, C]
            
        else:
            # --- Standard MLP-based processing ---
            x_mlp = x.permute(0, 2, 1)  # [B, C, seq_len]
            x_combined = torch.cat([x_mlp, latent], dim=-1)  # Concatenate with latent features
            shared_features = self.shared_module(x_combined)  # [B, C, shared_dim]
            
            trend = torch.relu(self.trend_fc1(shared_features))
            trend = torch.relu(self.trend_fc2(trend))
            trend = torch.relu(self.trend_fc3(trend))
            trend = torch.relu(self.trend_fc4(trend))
            trend = self.trend_fc5(trend)
            
            residual = torch.relu(self.residual_fc1(shared_features))
            residual = torch.relu(self.residual_fc2(residual))
            residual = torch.relu(self.residual_fc3(residual))
            residual = torch.relu(self.residual_fc4(residual))
            residual = self.residual_fc5(residual)
            
            out = trend + residual
            out = out.permute(0, 2, 1)
            
        return out