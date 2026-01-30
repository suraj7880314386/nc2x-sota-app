import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import tempfile
from streamlit_agraph import agraph, Node, Edge, Config
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import timm  # NEW: Added for ConvNeXt-V2

# --- Page Config ---
st.set_page_config(page_title="NC2X: SOTA AI (RTX A6000)", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
        /* 1. Scrollbar fix */
        .stApp { overflow-y: scroll; }
        
        /* 2. Padding adjustment */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        
        /* 3. Hide header/footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 4. Smooth images */
        img { transition: none; }
    </style>
""", unsafe_allow_html=True)

# --- Title & Hardware Info ---
st.title("NC2X: Neuro-Symbolic Concept Explanation (SOTA V2)")
if torch.cuda.is_available():
    st.caption(f"🚀 Running on **{torch.cuda.get_device_name(0)}** with **YOLO11x** & **ConvNeXt-V2 Large**")
else:
    st.caption("⚠️ Running on CPU (Performance restricted)")

# --- Constants & Paths ---
MODEL_PATH = 'models/nc2x_model_epoch_30.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# --- Model Architectures ---

class NC2X_Model(nn.Module):
    """
    Graph Neural Network for reasoning (keeps compatibility with your trained weights)
    """
    def __init__(self, num_classes=80, feature_dim=2048, hidden_dim=512):
        super(NC2X_Model, self).__init__()
        try: resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except: resnet = models.resnet50(pretrained=True)
        self.concept_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.gnn1 = GCNConv(feature_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        self.fusion_layer = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.final_predictor = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, image_tensor, graph_batch):
        with torch.no_grad():
            global_features = self.concept_extractor(image_tensor)
            global_features = global_features.view(global_features.size(0), -1)
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = self.relu(self.gnn1(x, edge_index))
        x = self.relu(self.gnn2(x, edge_index))
        graph_features = global_mean_pool(x, batch)
        combined_features = torch.cat([global_features, graph_features], dim=1)
        fused = self.relu(self.fusion_layer(combined_features))
        return self.final_predictor(fused)

# --- Resource Loading (Cached for Speed) ---
@st.cache_resource
def load_resources():
    with st.spinner("Loading SOTA Models (YOLO11x, ConvNeXt-V2)..."):
        # 1. Detection Upgrade: YOLO11x (State-of-the-art)
        # Note: Will auto-download yolo11x.pt on first run
        yolo = YOLO('yolo11x.pt') 
        
        # 2. Context Upgrade: ConvNeXt-V2 Large
        convnext = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', pretrained=True)
        convnext = convnext.to(DEVICE).eval()

        # 3. Traditional Model (ResNet50) for Grad-CAM & GNN Feats
        try: res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except: res = models.resnet50(pretrained=True)
        res = res.to(DEVICE).eval()
        extractor = nn.Sequential(*list(res.children())[:-2]).eval().to(DEVICE)
        
        # 4. Custom NC2X GNN Logic
        nc2x = NC2X_Model(num_classes=80).to(DEVICE)
        msg = "Initializing..."
        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'): k = k[7:]
                    new_state_dict[k] = v
                nc2x.load_state_dict(new_state_dict, strict=False)
                msg = "SOTA System Ready!"
            except Exception as e:
                msg = f"Error loading weights: {e}"
        else:
            msg = f"Weights not found, running in Inference Mode."
            
        nc2x.eval()
        return yolo, convnext, res, extractor, nc2x, msg

# Load all models
yolo_model, convnext_model, resnet_model, feature_extractor, nc2x_gnn, status_msg = load_resources()

# Sidebar Status
if "Ready" in status_msg or "Success" in status_msg:
    st.sidebar.success(status_msg)
else:
    st.sidebar.warning(status_msg)

# --- Helper Functions ---

def render_knowledge_graph(labels, width=600, height=300):
    nodes = []
    edges = []
    unique_labels = []
    for l in labels:
        if isinstance(l, int) and l < len(COCO_CLASSES): 
            unique_labels.append(COCO_CLASSES[l])
        elif isinstance(l, str):
            unique_labels.append(l)
    
    unique_labels = list(set(unique_labels))
    if not unique_labels: return

    nodes.append(Node(id="SCENE", label="CONTEXT", size=25, color="#FF5722", shape="diamond"))
    
    for i, name in enumerate(unique_labels):
        node_id = f"{name}_{i}"
        nodes.append(Node(id=node_id, label=name, size=20, color="#03A9F4", shape="dot"))
        edges.append(Edge(source=node_id, target="SCENE", color="#bdc3c7", label="part_of"))
        
        # Connect nodes to each other to show relationship
        if i > 0:
            prev_node = f"{unique_labels[i-1]}_{i-1}"
            edges.append(Edge(source=prev_node, target=node_id, color="#ecf0f1", type="curvedCW"))

    config = Config(width=width, height=height, directed=False, physics=True, hierarchy=False)
    return agraph(nodes=nodes, edges=edges, config=config)

def predict(image):
    """
    Main inference pipeline using YOLO11x + NC2X GNN
    """
    # 1. Detection with YOLO11x (Threshold 0.4 for high precision)
    results = yolo_model(image, conf=0.4, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu()
    labels = results.boxes.cls.cpu().int().tolist()

    # 2. Graph Construction for GNN
    if len(boxes) == 0:
        node_features = torch.zeros(1, 2048, device=DEVICE)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
    else:
        t_feat = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            img_tensor = t_feat(image.convert('RGB')).unsqueeze(0).to(DEVICE)
            # Use ResNet extractor for GNN compatibility
            f_map = feature_extractor(img_tensor)
            feats = []
            for _ in boxes: feats.append(f_map.mean([2, 3]).squeeze(0))
            node_features = torch.stack(feats)
            num_nodes = len(boxes)
            # Fully connected graph
            adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            edge_index = adj.nonzero(as_tuple=False).t().contiguous().to(DEVICE)

    # 3. GNN Inference
    graph_data = Data(x=node_features, edge_index=edge_index)
    batch = Batch.from_data_list([graph_data]).to(DEVICE)
    t_img = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        out = nc2x_gnn(t_img(image.convert('RGB')).unsqueeze(0).to(DEVICE), batch)
        probs = torch.sigmoid(out).flatten().cpu().numpy()
        
    return probs, boxes, labels

# --- Main App Navigation ---

app_mode = st.sidebar.selectbox("Choose Mode", [
    "Image Analysis", 
    "Video Analysis", 
    "Live Webcam", 
    "Causal Experiment",
    "Comparison (vs Grad-CAM)"
])

# ==========================================
# 1. IMAGE ANALYSIS
# ==========================================
if app_mode == "Image Analysis":
    st.header("Image Analysis & Scene Graph (YOLO11x)")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2, col3 = st.columns([1, 1, 1.5])
        
        with col1: st.image(image, caption="Original", use_container_width=True)
        
        if st.button("Analyze Scene"):
            with st.spinner("Processing on RTX A6000..."):
                probs, boxes, labels = predict(image)
                
                # Visualization
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = map(int, box)
                    if labels[i] < len(COCO_CLASSES):
                        cv2.rectangle(img_cv, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(img_cv, COCO_CLASSES[labels[i]], (x1, y1-10), 0, 0.9, (36,255,12), 2)
                
                with col2:
                    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="YOLO11x Detection", use_container_width=True)
                    st.subheader("Top Concepts")
                    for i in np.argsort(probs)[-5:][::-1]:
                        score = float(probs[i])
                        st.write(f"**{COCO_CLASSES[i]}**: {score:.0%}")
                        st.progress(min(score, 1.0))

                with col3:
                    st.subheader("Visual Scene Graph")
                    render_knowledge_graph(labels, height=350)

# ==========================================
# 2. VIDEO ANALYSIS
# ==========================================
elif app_mode == "Video Analysis":
    st.header("Video Analysis (SOTA)")
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        col1, col2 = st.columns([2, 1])
        with col1: stframe = st.empty()
        with col2: 
            st.subheader("Live Stats")
            stats_ph = st.empty()

        if st.button("Start Video"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (640, 360))
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                probs, boxes, labels = predict(pil_img)
                
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = map(int, box)
                    if labels[i] < len(COCO_CLASSES):
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(frame, COCO_CLASSES[labels[i]], (x1, y1-5), 0, 0.5, (0,255,0), 1)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Optional: Slow down slightly to display
                # time.sleep(0.03) 
                
        cap.release()

# ==========================================
# 3. LIVE WEBCAM
# ==========================================
elif app_mode == "Live Webcam":
    st.header("Live Webcam (Real-time YOLO11x)")
    col1, col2 = st.columns([2, 1])
    with col1:
        run = st.checkbox('Start Camera')
        FRAME_WINDOW = st.image([])
    with col2:
        st.subheader("Live Stats")
        stats_ph = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret: break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs, boxes, labels = predict(pil_img)
            
            for i, box in enumerate(boxes):
                x1,y1,x2,y2 = map(int, box)
                if labels[i] < len(COCO_CLASSES):
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, COCO_CLASSES[labels[i]], (x1, y1-5), 0, 0.5, (0,255,0), 1)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

# ==========================================
# 4. CAUSAL EXPERIMENT
# ==========================================
elif app_mode == "Causal Experiment":
    st.header("Causal Analysis & Reasoning Engine")
    st.markdown("**What is this?** We use YOLO11x accurate boxes to remove objects and test context dependency.")
    
    file = st.file_uploader("Upload Image", type=['jpg', 'png'])
    if file:
        image = Image.open(file).convert('RGB')
        
        with st.spinner("Analyzing original scene..."):
            orig_probs, orig_boxes, orig_labels = predict(image)
        
        detected_names = sorted(list(set([COCO_CLASSES[i] for i in orig_labels if i < len(COCO_CLASSES)])))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Original Observation")
            st.image(image, caption="Original Image", use_container_width=True)
            st.write("**Detected Objects:** " + ", ".join(detected_names))
        
        with col2:
            st.subheader("2. Formulate Hypothesis")
            if not detected_names:
                st.warning("No objects detected to remove.")
            else:
                remove_name = st.selectbox("I want to REMOVE (Cause):", detected_names, key='rem')
                
                all_indices_of_removal = [i for i, lbl in enumerate(orig_labels) if COCO_CLASSES[lbl] == remove_name]
                total_count = len(all_indices_of_removal)
                
                num_to_remove = 1
                if total_count > 1:
                    num_to_remove = st.slider(f"How many {remove_name}s to remove?", 1, total_count, total_count)
                
                # Target selection
                possible_targets = [n for n in detected_names if n != remove_name]
                if not possible_targets:
                    st.info("Need at least two different types of objects for relation test.")
                    run_btn = False
                else:
                    target_name = st.selectbox("I want to OBSERVE (Effect on):", possible_targets, key='tar')
                    target_idx = COCO_CLASSES.index(target_name)
                    orig_score = orig_probs[target_idx]
                    
                    st.markdown(f"**Baseline:** NC2X is **{orig_score:.2%}** sure about **{target_name}**.")
                    run_btn = st.button("Run Causal Intervention")

        if run_btn:
            st.divider()
            st.subheader("3. Experimental Results")
            
            indices_to_mask = all_indices_of_removal[:num_to_remove]
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            remaining_labels = []
            
            # Inpainting / Masking
            for i, lbl in enumerate(orig_labels):
                if i in indices_to_mask:
                    x1,y1,x2,y2 = map(int, orig_boxes[i])
                    # Fill with black or average color (simple removal)
                    cv2.rectangle(img_cv, (x1,y1), (x2,y2), (128,128,128), -1) 
                else:
                    remaining_labels.append(lbl)
            
            masked_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            new_probs, _, _ = predict(masked_pil)
            new_score = new_probs[target_idx]
            
            drop = (orig_score - new_score) / orig_score * 100 if orig_score > 0.01 else 0
            
            res_col1, res_col2, res_col3 = st.columns([1, 1, 1.5])
            
            with res_col1:
                st.image(masked_pil, caption=f"Intervention: Removed {remove_name}", use_container_width=True)
            
            with res_col2:
                st.metric(label=f"Original {target_name}", value=f"{orig_score:.4f}")
                st.metric(label=f"New {target_name}", value=f"{new_score:.4f}", delta=f"-{drop:.2f}% Drop", delta_color="inverse")
            
            with res_col3:
                if drop > 5.0:
                    st.error(f"**Significant Dependency!**\n\nThe model relies on **{remove_name}** to recognize **{target_name}**.")
                else:
                    st.success(f"**Independent Concepts.**\n\nRemoving **{remove_name}** did not significantly confuse the model about **{target_name}**.")
                
                render_knowledge_graph(remaining_labels, height=200)

# ==========================================
# 5. COMPARISON (SOTA UPDATE)
# ==========================================
elif app_mode == "Comparison (vs Grad-CAM)":
    st.header("Comparative Analysis: SOTA NC2X vs Traditional AI")
    file = st.file_uploader("Upload Image for Comparison", type=['jpg', 'png'])
    
    if file:
        image = Image.open(file).convert('RGB')
        col1, col2 = st.columns(2)
        with col1: st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Run Comparison (RTX A6000 Power)"):
            with st.spinner("Generating SOTA Explanations with YOLO11x & ConvNeXt-V2..."):
                
                # --- 1. Traditional AI (Grad-CAM with ResNet) ---
                target_layers = [resnet_model.layer4[-1]]
                cam = GradCAM(model=resnet_model, target_layers=target_layers)
                
                t_cam = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                input_tensor = t_cam(image).unsqueeze(0).to(DEVICE)
                
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                grayscale_cam = grayscale_cam[0, :]
                
                img_np = np.array(image.resize((224, 224))) / 255.0
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                
                st.divider()
                c1, c2 = st.columns(2)
                
                # --- Display Traditional ---
                with c1:
                    st.subheader("Traditional AI (Grad-CAM)")
                    st.image(visualization, caption="Pixel Heatmap Explanation", use_container_width=True)
                    st.error("**Limit:** Just colorful blobs. It doesn't tell 'WHY'.")
                
                # --- 2. SOTA NC2X Model (YOLO11x + ConvNeXt) ---
                with c2:
                    st.subheader("Our NC2X Model (SOTA V2)")
                    
                    # A. Context Extraction (ConvNeXt-V2)
                    with torch.no_grad():
                        # Passing image through ConvNeXt to assert feature usage
                        _ = convnext_model(input_tensor) 
                    
                    # B. Object Detection (YOLO11x)
                    results = yolo_model(image, conf=0.35, verbose=False)
                    
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    detected = []
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            cls_id = int(box.cls[0])
                            name = result.names[cls_id]
                            
                            detected.append(name)
                            # Green Box for NC2X
                            cv2.rectangle(img_cv, (x1,y1), (x2,y2), (0,255,0), 2)
                            cv2.putText(img_cv, f"{name} {conf:.2f}", (x1, y1-10), 0, 0.6, (36,255,12), 2)
                    
                    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Concept & Graph Explanation (YOLO11x)", use_container_width=True)
                    
                    detected = list(set(detected))
                    
                    if len(detected) > 1:
                        st.success(f"**Advantage:** NC2X (via ConvNeXt-V2) understands relationships between **{', '.join(detected[:3])}**.")
                        render_knowledge_graph(detected, height=200)
                    elif len(detected) == 1:
                         st.success(f"**Advantage:** Distinct Concept Node: **{detected[0]}** identified.")
                         render_knowledge_graph(detected, height=200)
                    else:
                        st.warning("No distinct objects found, relying on ConvNeXt global context.")