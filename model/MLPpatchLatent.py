import torch
import torch.nn as nn
import pandas as pd
from kalman_filter import apply_kalman_filter

class KalmanDecomposition(nn.Module):
    def __init__(self, observation_covariance=1, transition_covariance=1):
        super(KalmanDecomposition, self).__init__()
        self.obs_cov = observation_covariance
        self.trans_cov = transition_covariance
        
    def forward(self, obs):
        """_summary_

        Args:
            obs : Tensor of shape [B, seq_len, C]

        Returns:
            trend: Tensor of shape [B, seq_len, C]
            residual: Tensor of shape [B, seq_len, C]
        """
        B, L, C = obs.shape
        trend = torch.zeros_like(obs) # Secure memory of size [B,L,C]
        residual = torch.zeros_like(obs)
        for b in range(B):
            for c in range(C):
                # Tensor to Numpy
                series_tensor = obs[b, :, c].detach().cpu()
                series_numpy = pd.Series(series_tensor.numpy(), index=range(L))
                smoothed_series = apply_kalman_filter(
                    series_numpy, self.obs_cov, self.trans_cov
                )
                # Numpy to Tensor
                trend[b, :, c] = torch.tensor(smoothed_series.values, device=obs.device, dtype=obs.dtype)
                residual[b, :, c] = obs[b, :, c] - trend[b, :, c]
        return trend, residual

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len       # Input sequence length
        self.pred_len = configs.pred_len     # Prediction sequence length
        self.c_in = configs.enc_in           # Number of channels
        
        # Patching with stride
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.padding_patch = configs.padding_patch
        
        self.obs_cov = configs.obs_cov # observation covariance of Kalman Filter
        self.trans_cov = configs.trans_cov # transition covariance of Kalman Filter
        
        # Settings for extracting latent features using Transformer
        self.d_model = getattr(configs, 'd_model', 64)   # Transformer embedding dimension
        self.nhead = getattr(configs, 'nhead', 4)          # Number of heads in multi-head attention
        self.num_layers = getattr(configs, 'num_layers', 3)  # Number of Transformer encoder layers
        self.latent_proj = nn.Linear(1, self.d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_layers)
        
        # Decomposition using Kalman Filter
        self.kalman_decomp = KalmanDecomposition(
            observation_covariance = self.obs_cov,
            transition_covariance = self.trans_cov)
        
        # --- Patch-based processing setup ---
        self.n_patches = (self.seq_len - self.patch_size) // self.patch_stride + 1 # Number of patch
        if self.padding_patch == 'end': # Apply Paddings
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))
            self.n_patches += 1

        
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
        x_trans = self.transformer_encoder(x_proj) # commpare
        # Perform mean pooling over the time dimension to generate latent vectors per channel
        latent = x_trans.mean(dim=0)  # [B*C, d_model] # commpare
        # Restore original batch and channel dimensions: [B, C, d_model]
        latent = latent.view(B, C, self.d_model)
        return latent
    
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, seq_len, C]
        Returns:
            out: Output tensor of shape [B, pred_len, C]
        
        Descriptions:

        """
        B, L, C = x.size()
        
        # Extract latent feature from Transformer / shape: [B, C, d_model]
        latent = self.compute_latent(x)
        
        # Decompose using Kalman Filter / shape: [B, seq_len, C]
        trend_obs, residual_obs = self.kalman_decomp(x)
        
        # ----- Trend branch -----
        # Patch extraction from trend_obs 
        x_trend = trend_obs.permute(0, 2, 1)  # [B, C, seq_len]
        if self.padding_patch == 'end':
            x_trend = self.padding_patch_layer(x_trend)
        trend_patches = x_trend.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        trend_patch_emb = self.patch_embedding(trend_patches)  # [B, C, n_patches, 128]
        trend_patch_emb = torch.relu(trend_patch_emb)
        trend_patch_emb = trend_patch_emb.view(B, C, -1)  # [B, C, 128*n_patches]
        
        # ----- Residual branch -----
        # Patch extraction from residual_obs
        x_resid = residual_obs.permute(0, 2, 1)  # [B, C, seq_len]
        if self.padding_patch == 'end':
            x_resid = self.padding_patch_layer(x_resid)
        resid_patches = x_resid.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        resid_patch_emb = self.patch_embedding(resid_patches)  # [B, C, n_patches, 128]
        resid_patch_emb = torch.relu(resid_patch_emb)
        resid_patch_emb = resid_patch_emb.view(B, C, -1)  # [B, C, 128*n_patches]
        
        # Combine latent with each stream 
        trend_combined = torch.cat([trend_patch_emb, latent], dim=-1)  # [B, C, shared_dim]
        resid_combined = torch.cat([resid_patch_emb, latent], dim=-1)  # [B, C, shared_dim]
        
        # Shared module 
        trend_features = self.shared_module(trend_combined)  # [B, C, shared_dim]
        resid_features = self.shared_module(resid_combined)  # [B, C, shared_dim]
        
        # Predict using MLP
        trend_pred = self.trend_aggregator(trend_features)    # [B, C, pred_len]
        resid_pred = self.residual_aggregator(resid_features)   # [B, C, pred_len]
        
        # Final Prediction (trend+resid)
        out = trend_pred + resid_pred  # [B, C, pred_len]
        out = out.permute(0, 2, 1)      # [B, pred_len, C]
        
        return out
    