import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, PointNetConv, knn_graph

class GNNModel(nn.Module):
    def __init__(self, model_type="dgcnn", in_channels=32, hidden_dim=64, k=20, out_channels=1):
        super().__init__()
        self.model_type = model_type
        self.k = k
        if model_type == "dgcnn":
            self.conv1 = DynamicEdgeConv(nn.Sequential(nn.Linear(2*in_channels, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()), k, aggr='max')
            self.conv2 = DynamicEdgeConv(nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()), k, aggr='max')
        elif model_type == "pointnet": # Basic PointNetConv
            self.conv1 = PointNetConv(nn.Sequential(nn.Linear(in_channels+3, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()))
            self.conv2 = PointNetConv(nn.Sequential(nn.Linear(hidden_dim+3, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, data, pretrain=False):
        x, pos, batch = data.x, data.pos, data.batch
        if self.model_type == "dgcnn":
            x = self.conv1(x, batch)
            x = self.conv2(x, batch)
        else:
            edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
            x = self.conv1(x, pos, edge_index)
            x = self.conv2(x, pos, edge_index)
        out = self.classifier(x)
        if pretrain:
            return out
        else:
            return out.squeeze(-1) if out.shape[-1] == 1 else out
