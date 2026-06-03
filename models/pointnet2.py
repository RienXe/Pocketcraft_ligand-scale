import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, knn_interpolate
from torch_cluster import radius

class SAModule(torch.nn.Module):
    """
    Set Abstraction (SA) Module.
    
    Downsamples the point cloud using Farthest Point Sampling (FPS) and aggregates 
    local features within a specified radius using PointNet layers.
    """
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio) # Farthest Point Sampling
        # radius query finds all points in pos within radius r around pos[idx]
        assign_index = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        row, col = assign_index[0], assign_index[1]
        edge_index = torch.stack([col, row], dim=0) # source -> target
        
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class FPModule(torch.nn.Module):
    """
    Feature Propagation (FP) Module.
    
    Upsamples the point cloud features from a downsampled level back to a 
    higher-resolution level using k-nearest neighbor (k-NN) interpolation 
    and skip connections.
    """
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1) # Skip connection
        x = self.nn(x)
        return x, pos_skip, batch_skip

class PointNet2Segmenter(nn.Module):
    """
    U-Net style PointNet++ Model tailored for point-wise prediction.
    Features specific hyperparameters: radii (r), sampling ratios, k, and dropout.
    """
    def __init__(self, in_channels, out_channels, hidden_dim=64,
                 sa1_ratio=0.25, sa1_r=8.0,
                 sa2_ratio=0.25, sa2_r=16.0,
                 fp_k=3, dropout=0.3):
        super().__init__()
        
        # Encoding path (Set Abstraction)
        self.sa1 = SAModule(sa1_ratio, sa1_r, MLP([in_channels + 3, hidden_dim, hidden_dim]))
        self.sa2 = SAModule(sa2_ratio, sa2_r, MLP([hidden_dim + 3, hidden_dim * 2, hidden_dim * 2]))
        
        # Decoding path (Feature Propagation)
        self.fp2 = FPModule(fp_k, MLP([hidden_dim * 2 + hidden_dim, hidden_dim, hidden_dim]))
        self.fp1 = FPModule(fp_k, MLP([hidden_dim + in_channels, hidden_dim, hidden_dim]))
        
        # Main Prediction Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels)
        )
        
        # Pretraining Head (Feature Reconstruction)
        self.pretrain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output dimension matches input feature dimension
            nn.Linear(hidden_dim, in_channels)
        )

    def forward(self, data, pretrain=False):
        x0, pos0, batch0 = data.x, data.pos, data.batch
        
        # Encoder
        x1, pos1, batch1 = self.sa1(x0, pos0, batch0)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)

        # Decoder
        x_fp2, pos_fp2, batch_fp2 = self.fp2(x2, pos2, batch2, x1, pos1, batch1)

        x_fp1, pos_fp1, batch_fp1 = self.fp1(x_fp2, pos_fp2, batch_fp2, x0, pos0, batch0)
        
        if pretrain:
            return self.pretrain_head(x_fp1)
        else:
            out = self.classifier(x_fp1)
            return out.squeeze(-1) if out.shape[-1] == 1 else out
