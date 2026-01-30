import os, torch, timm, json
from pycocotools.coco import COCO
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from torch_geometric.data import Data
from tqdm import tqdm


DEVICE = 'cuda'
DATA_ROOT = '../data/coco/val2017'
ANN_FILE = '../data/coco/annotations/instances_val2017.json'
SAVE_DIR = '../data/coco/processed_val_data'
os.makedirs(SAVE_DIR, exist_ok=True)


yolo = YOLO('yolo11x.pt').to(DEVICE)
backbone = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0, global_pool='avg').eval().to(DEVICE)

coco = COCO(ANN_FILE)
ids = list(coco.imgs.keys())
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

print(f"Pre-calculating features for {len(ids)} Validation images...")

with torch.no_grad():
    for img_id in tqdm(ids):
        save_path = os.path.join(SAVE_DIR, f"{img_id}.pt")
        if os.path.exists(save_path): continue
        
        info = coco.loadImgs(img_id)[0]
        try: 
            img = Image.open(os.path.join(DATA_ROOT, info['file_name'])).convert('RGB')
        except: continue
        
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        global_x = backbone(img_t).cpu().squeeze(0)
        
        results = yolo(img, verbose=False, conf=0.3)[0]
        boxes = results.boxes.xyxy.cpu()
        
        if len(boxes) == 0:
            node_x, edge_index = torch.zeros(1, 1536), torch.empty((2, 0), dtype=torch.long)
        else:
            node_feats = []
            for box in boxes:

                crop_img = img.crop(tuple(map(int, box)))
                crop_tensor = transform(crop_img).unsqueeze(0).to(DEVICE)
                node_feats.append(backbone(crop_tensor).cpu().squeeze(0))
            
            node_x = torch.stack(node_feats)
            adj = torch.ones(len(boxes), len(boxes)) - torch.eye(len(boxes))
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            
        
        cat_ids = coco.getCatIds()
        target = torch.zeros(len(cat_ids), dtype=torch.float32)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        target[[cat_ids.index(a['category_id']) for a in anns if a['category_id'] in cat_ids]] = 1.0
        torch.save({'global_x': global_x, 'node_x': node_x, 'edge_index': edge_index, 'y': target}, save_path)
print("✅ Validation Preprocessing Complete!")
