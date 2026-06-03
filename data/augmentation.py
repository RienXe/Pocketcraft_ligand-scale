import torch
import math

def augment_point_cloud(pos, jitter_val=0.02):
    """Applies random rotation and translation to the point cloud."""
    # 1. Random Rotation (around Z-axis)
    theta = torch.rand(1) * 2 * math.pi
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    pos = torch.matmul(pos, R)
    
    # 2. Random Translation (Small shift in space)
    translation = (torch.rand(1, 3) - 0.5) * 1.0 # Max 0.5A shift in each direction
    pos = pos + translation
    
    # 3. Slight Jitter (Point-wise noise)
    jitter = torch.randn_like(pos) * jitter_val
    return pos + jitter
