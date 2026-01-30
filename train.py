import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import os, glob

from dataset import FastNC2XDataset
from model import NC2X_TrainerModel

# --- A6000 ULTRA CONFIG ---
# Kyunki sirf GNN train ho raha hai, hum BATCH_SIZE 512+ kar sakte hain!
BATCH_SIZE = 512       
EPOCHS = 100
LEARNING_RATE = 1e-3   
DEVICE = 'cuda'
SAVE_DIR = '../data/output'
PROCESSED_DIR = '../data/coco/processed_data'
os.makedirs(SAVE_DIR, exist_ok=True)

def collate_fn(batch):
    gx, graphs, y = zip(*batch)
    return torch.stack(gx), Batch.from_data_list(graphs), torch.stack(y)

def train():
    print(f"🚀 Training on {DEVICE} with Batch Size: {BATCH_SIZE}")
    
    dataset = FastNC2XDataset(data_dir=PROCESSED_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = NC2X_TrainerModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda') # For speed

    # --- 🔄 AUTO-RESUME LOGIC ---
    start_epoch = 0
    existing_models = glob.glob(f"{SAVE_DIR}/nc2x_sota_epoch_*.pth")
    if existing_models:
        latest_model = max(existing_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_model.split('_')[-1].split('.')[0])
        print(f"✅ Resuming from Epoch {start_epoch}")
        model.load_state_dict(torch.load(latest_model, weights_only=True))

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for gx, g, y in loop:
            gx, g, y = gx.to(DEVICE), g.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                out = model(gx, g)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Save Checkpoint
        save_path = f"{SAVE_DIR}/nc2x_sota_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved: {save_path}")

if __name__ == "__main__":
    train()