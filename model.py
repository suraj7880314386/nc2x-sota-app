import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class NC2X_TrainerModel(nn.Module):
    def __init__(self, num_classes=80, input_dim=1536, hidden_dim=1024):
        super(NC2X_TrainerModel, self).__init__()
        
        # 1. Contextualizer (GNN) - Yahi train hoga
        self.gnn1 = GCNConv(input_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

        # 2. Fusion Path (Global Features + Graph Context)
        # input_dim(1536) + hidden_dim(1024) = 2560
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, global_x, graph_batch):
        # Graph nodes processing
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = self.relu(self.gnn1(x, edge_index))
        x = self.relu(self.gnn2(x, edge_index))
        
        # Pooled graph features
        graph_context = global_mean_pool(x, batch)

        # Combine with pre-extracted global features
        combined = torch.cat([global_x, graph_context], dim=1)
        return self.fusion(combined)