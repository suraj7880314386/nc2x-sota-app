import torch, numpy as np, os, glob
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from dataset import FastNC2XDataset, collate_fn
from model import NC2X_TrainerModel

def evaluate():
    DEVICE = 'cuda'
    MODEL_PATH = '../data/output/nc2x_sota_epoch_100.pth'
    VAL_DIR = '../data/coco/processed_val_data'
    
    dataset = FastNC2XDataset(VAL_DIR)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    model = NC2X_TrainerModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for gx, g, y in tqdm(loader, desc="Testing"):
            gx, g = gx.to(DEVICE), g.to(DEVICE)
            out = torch.sigmoid(model(gx, g)).cpu().numpy()
            all_preds.append((out > 0.5).astype(int))
            all_targets.append(y.numpy().astype(int))
            
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    
    print("\n" + "="*30)
    print("NC2X V2 FINAL METRICS")
    print("="*30)
    print(f"Precision: {precision_score(y_true, y_pred, average='samples', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='samples', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='samples', zero_division=0):.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
