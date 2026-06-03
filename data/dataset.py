import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import scatter
from data.augmentation import augment_point_cloud

class ProteinPocketDataset(Dataset):
    def __init__(self, root_dir, files=None, is_train=True, voxel_size=None):
        super().__init__(root_dir, None, None)
        if files is None:
            self.files = [f for f in os.listdir(root_dir) if f.endswith('.pt')]
        else:
            self.files = [f for f in files if os.path.exists(os.path.join(root_dir, f))]
        self.root_dir = root_dir
        self.is_train = is_train
        self.voxel_size = voxel_size

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(os.path.join(self.root_dir, self.files[idx]))
        if isinstance(data, dict):
            data = Data(**data)
            
        # Optional Voxel Downsampling (to accelerate tuning)
        if self.voxel_size is not None:
            cluster = voxel_grid(data.pos, size=self.voxel_size)
            data.pos = scatter(data.pos, cluster, dim=0, reduce='mean')
            data.x = scatter(data.x, cluster, dim=0, reduce='mean')
            if hasattr(data, 'y') and data.y is not None:
                # Majority vote for binary labels
                data.y = (scatter(data.y.float(), cluster, dim=0, reduce='mean') > 0.5).long()
            if hasattr(data, 'batch'):
                 data.batch = torch.zeros(data.pos.size(0), dtype=torch.long)

        if self.is_train:
            data.pos = augment_point_cloud(data.pos)
        return data
