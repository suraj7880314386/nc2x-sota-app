import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class FastNC2XDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Saari saved files ki list
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        print(f"Dataset initialized with {len(self.file_list)} pre-processed samples.")

    def __getitem__(self, index):
        # Saved .pt file load karein
        file_path = os.path.join(self.data_dir, self.file_list[index])
        data_bundle = torch.load(file_path, weights_only=True)
        
        # graph_data: Nodes aur Edges (GNN ke liye)
        graph_data = Data(x=data_bundle['node_x'], edge_index=data_bundle['edge_index'])
        
        # Hum return karenge: Global Feature, Graph, aur Target
        return data_bundle['global_x'], graph_data, data_bundle['y']

    def __len__(self):
        return len(self.file_list)

# === YEH FUNCTION ADD KIYA GAYA HAI TAAKI evaluate.py ISTEMAAL KAR SAKE ===
def collate_fn(batch):
    # None values ko filter karein
    batch = [b for b in batch if b is not None]
    if not batch: return None, None, None
    
    global_x, graphs, targets = zip(*batch)
    
    # Global features ko stack karein
    global_x_batch = torch.stack(global_x)
    
    # Graphs ko PyG Batch format me badlein
    graph_batch = Batch.from_data_list(graphs)
    
    # Targets ko stack karein
    target_batch = torch.stack(targets)
    
    return global_x_batch, graph_batch, target_batch