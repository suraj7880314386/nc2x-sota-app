import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
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
import timm
import graphviz
import av

st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            background-color: #1E1E1E;
        }
        .st-emotion-cache-hp888p { 
            font-size: 25px; 
            color: #FF5722 !important;
            font-weight: bold;
        }
        .sidebar-hint {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 999;
            background: #FF5722;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
        }
    </style>
    <div class="sidebar-hint">Tip: Use the arrow at the top-left to toggle the menu</div>
""", unsafe_allow_html=True)

st.title("NC2X: Concept, Causal & Context-Aware AI")

MODEL_PATH = 'nc2x_sota_epoch_100.pth'
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

class NC2X_Model(nn.Module):
    def __init__(self, num_classes=80, feature_dim=1536, hidden_dim=1024):
        super(NC2X_Model, self).__init__()
        self.gnn1 = GCNConv(feature_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, global_x, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = self.relu(self.gnn1(x, edge_index))
        x = self.relu(self.gnn2(x, edge_index))
        graph_features = global_mean_pool(x, batch)
        combined = torch.cat([global_x, graph_features], dim=1)
        return self.fusion(combined)

@st.cache_resource
def load_resources():
    yolo = YOLO('yolo11x.pt')
    backbone = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k', 
                                 pretrained=True, num_classes=0, global_pool='avg')
    backbone = backbone.to(DEVICE).eval()
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    nc2x = NC2X_Model(num_classes=80, feature_dim=1536, hidden_dim=1024).to(DEVICE)
    
    msg = "System Ready"
    if os.path.exists(MODEL_PATH):
        try:
            nc2x.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            msg = "NC2X System Ready!"
        except Exception as e:
            msg = f"Weights Error: {e}"
    else:
        msg = "Running in Zero-shot Mode (Weights Missing)"
    
    nc2x.eval()
    return yolo, backbone, resnet, nc2x, msg

yolo, backbone, resnet_comp, nc2x_model, status_msg = load_resources()

st.sidebar.success(status_msg)
if torch.cuda.is_available():
    st.sidebar.info(f"Hardware: {torch.cuda.get_device_name(0)}")

def render_knowledge_graph(labels, width=600, height=300):
    nodes, edges = [], []
    unique_labels = list(set([COCO_CLASSES[l] if isinstance(l, int) else l for l in labels]))
    if not unique_labels: return
    
    nodes.append(Node(id="SCENE", label="CONTEXT", size=25, color="#FF5722", shape="diamond"))
    for name in unique_labels:
        nodes.append(Node(id=name, label=name, size=20, color="#03A9F4"))
        edges.append(Edge(source=name, target="SCENE", color="#bdc3c7"))
    
    config = Config(width=width, height=height, directed=False, physics=True)
    return agraph(nodes=nodes, edges=edges, config=config)

def predict(image):
    results = yolo(image, conf=0.4, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu()
    labels = results.boxes.cls.cpu().int().tolist()
    
    t_feat = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        img_t = t_feat(image.convert('RGB')).unsqueeze(0).to(DEVICE)
        global_x = backbone(img_t)
        
        if len(boxes) == 0:
            node_features = torch.zeros(1, 1536, device=DEVICE)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
        else:
            node_feats = []
            for box in boxes:
                crop = image.crop(tuple(map(int, box)))
                node_feats.append(backbone(t_feat(crop).unsqueeze(0).to(DEVICE)).squeeze(0))
            node_features = torch.stack(node_feats)
            num_nodes = len(boxes)
            adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            edge_index = adj.nonzero(as_tuple=False).t().contiguous().to(DEVICE)
            
        batch = Batch.from_data_list([Data(x=node_features, edge_index=edge_index)]).to(DEVICE)
        out = nc2x_model(global_x, batch)
        probs = torch.sigmoid(out).flatten().cpu().numpy()
        return probs, boxes, labels

class SOTAVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detected_items = []
        self.frame_count = 0

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        if self.frame_count % 10 == 0:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            probs, boxes, labels = predict(pil_img)
            
            current_detected = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                name = COCO_CLASSES[labels[i]]
                current_detected.append(name)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            
            self.detected_items = list(set(current_detected))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

app_mode = st.sidebar.selectbox("Choose Mode", [
    "Image Analysis", "Video Analysis", "Live Webcam", "Causal Experiment", "Comparison (vs Grad-CAM)"
])

if app_mode == "Image Analysis":
    st.header("Scene Graph Analysis")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2, col3 = st.columns([1, 1, 1.2])
        with col1: st.image(image, caption="Original", use_container_width=True)
        
        if st.button("Analyze Scene"):
            with st.spinner("Processing..."):
                probs, boxes, labels = predict(image)
                img_draw = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = map(int, box)
                    cv2.rectangle(img_draw, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img_draw, COCO_CLASSES[labels[i]], (x1, y1-10), 0, 0.8, (36,255,12), 2)
                
                with col2:
                    st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), caption="YOLOv11 Analysis", use_container_width=True)
                    st.subheader("Detected Concepts")
                    for i in np.argsort(probs)[-5:][::-1]:
                        st.write(f"**{COCO_CLASSES[i]}**: {probs[i]:.0%}")
                with col3:
                    st.subheader("Knowledge Graph")
                    render_knowledge_graph(labels)

elif app_mode == "Video Analysis":
    st.header("Video Processing")
    st.info("Performance Note: Processing every 5th frame for optimization.")
    v_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    
    if v_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(v_file.read())
        tfile.close()
        
        if st.button("Start Processing"):
            col_left, col_right = st.columns([2, 1])
            with col_left: st_frame = st.empty()
            with col_right:
                st.subheader("Active Detections")
                st_list = st.empty()
            
            st_status = st.info("Initializing analysis engine...")
            cap = cv2.VideoCapture(tfile.name)
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % 5 == 0:
                    frame = cv2.resize(frame, (480, 270))
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    _, boxes, labels = predict(pil_img)
                    
                    current_frame_objects = []
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        obj_name = COCO_CLASSES[labels[i]]
                        current_frame_objects.append(obj_name)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, obj_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Processing Frame: {frame_idx}")
                    unique_objects = list(set(current_frame_objects))
                    if unique_objects:
                        obj_list_text = "\n".join([f" **{obj}**" for obj in unique_objects])
                        st_list.markdown(obj_list_text)
                    else:
                        st_list.write("Scanning frame...")
                frame_idx += 1
            cap.release()
            st_status.success("Video analysis complete.")
            os.unlink(tfile.name)

elif app_mode == "Live Webcam":
    st.header("Live Symbolic Inference")
    st.info("Snapshot mode is utilized for stability when running complex models.")

    col_v, col_l = st.columns([1.5, 1])

    with col_v:
        img_file = st.camera_input("Take a snapshot for analysis")

    with col_l:
        st.subheader("Detection Results")
        if img_file:
            with st.spinner("Analyzing snapshot..."):
                input_img = Image.open(img_file).convert('RGB')
                probs, boxes, labels = predict(input_img)
                
                for lbl in labels:
                    st.success(f"Detected: **{COCO_CLASSES[lbl]}**")
                
                st.write("---")
                st.write("**Knowledge Graph:**")
                render_knowledge_graph(labels, height=250)
        else:
            st.warning("Please capture a snapshot to trigger symbolic reasoning.")

elif app_mode == "Causal Experiment":
    st.header("Causal Reasoning & Intervention Engine")
    
    st.info("""
    **Methodology:** We are testing the model's robustness by manually removing symbolic nodes (objects). 
    This allows us to observe how much the AI relies on context vs. direct visual features.
    """)
    
    file = st.file_uploader("Upload Image to Begin Experiment", type=['jpg', 'png', 'jpeg'])
    
    if file:
        image = Image.open(file).convert('RGB')
        
        if 'causal_data' not in st.session_state:
            st.subheader("Step 1: Baseline Observation")
            if st.button("Run Initial Scene Analysis"):
                with st.spinner("Performing baseline analysis..."):
                    orig_probs, orig_boxes, orig_labels = predict(image)
                    st.session_state.causal_data = {
                        'probs': orig_probs,
                        'boxes': orig_boxes,
                        'labels': orig_labels,
                        'names': sorted(list(set([COCO_CLASSES[i] for i in orig_labels if i < len(COCO_CLASSES)])))
                    }
                    st.rerun()
        
        if 'causal_data' in st.session_state:
            data = st.session_state.causal_data
            col_img, col_setup = st.columns([1.2, 1])
            
            with col_img:
                st.image(image, caption="Reference Baseline", use_container_width=True)
                st.write(f"**Identified Concepts:** {', '.join(data['names'])}")

            with col_setup:
                st.subheader("Step 2: Define Hypothesis")
                remove_name = st.selectbox("Select object to REMOVE (Cause):", data['names'], key='rem_obj')
                
                all_idx = [i for i, lbl in enumerate(data['labels']) if COCO_CLASSES[lbl] == remove_name]
                total_count = len(all_idx)
                
                num_to_remove = st.slider(f"Quantity of '{remove_name}' to hide:", 1, total_count, total_count)
                
                targets = [n for n in data['names'] if n != remove_name]
                target_name = st.selectbox("Observe impact on (Effect):", targets if targets else ["No other objects"], key='tar_obj')
                
                if target_name != "No other objects":
                    target_idx = COCO_CLASSES.index(target_name)
                    orig_score = data['probs'][target_idx]
                    
                    st.markdown(f"**Hypothesis:** If I hide **{num_to_remove} {remove_name}(s)**, the confidence in identifying the **{target_name}** will change.")
                    st.write(f"**Current Confidence:** `{orig_score:.4%}`")
                    
                    run_trigger = st.button("Run Causal Intervention")

            if 'run_trigger' in locals() and run_trigger:
                st.divider()
                st.subheader("Step 3: Intervention Results")
                
                with st.spinner("Calculating impact of intervention..."):
                    indices_to_mask = all_idx[:num_to_remove]
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    remaining_labels = []
                    
                    for i, lbl in enumerate(data['labels']):
                        if i in indices_to_mask:
                            x1, y1, x2, y2 = map(int, data['boxes'][i])
                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (15, 15, 15), -1)
                            cv2.putText(img_cv, "HIDDEN", (x1+5, y1+20), 0, 0.6, (255,255,255), 2)
                        else:
                            remaining_labels.append(lbl)
                    
                    intervened_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                    new_probs, _, _ = predict(intervened_img)
                    new_score = new_probs[target_idx]
                    
                    diff = orig_score - new_score
                    drop_percent = (diff / orig_score) * 100 if orig_score > 0 else 0
                    
                    res_c1, res_c2, res_c3 = st.columns([1.2, 1, 1.5])
                    
                    with res_c1:
                        st.image(intervened_img, caption="Modified Context", use_container_width=True)
                    
                    with res_c2:
                        st.metric(label=f"Baseline {target_name}", value=f"{orig_score:.2%}")
                        st.metric(label=f"Intervention Confidence", value=f"{new_score:.2%}", 
                                  delta=f"{drop_percent:+.4f}% Change", delta_color="inverse")
                    
                    with res_c3:
                        st.markdown("### Experimental Results")
                        st.info(f"Hiding {num_to_remove} {remove_name} nodes resulted in a {abs(drop_percent):.4f}% shift in recognition confidence for the target {target_name}.")

                        st.markdown("### Reasoning Logic")
                        if abs(drop_percent) < 0.01:
                            st.success(f"**Result: Model Robustness.** The NC2X model is stable. Identifying the {target_name} does not rely on the presence of {remove_name}.")
                        elif drop_percent > 2:
                            st.error(f"**Result: Contextual Dependency.** The AI uses {remove_name} as contextual proof to verify the identity of the {target_name}.")
                        else:
                            st.warning("Minor context dependence observed.")
                        
                        st.write("---")
                        st.write("**Updated Symbolic Graph:**")
                        render_knowledge_graph(remaining_labels)

            st.sidebar.button("Clear Experiment Data", on_click=lambda: st.session_state.pop('causal_data', None))

elif app_mode == "Comparison (vs Grad-CAM)":
    st.header("Grad-CAM vs NC2X Benchmark")
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        image = Image.open(file).convert('RGB')
        if st.button("Compare Architectures"):
            with st.spinner("Generating heatmaps..."):
                target_layers = [resnet_comp.layer4[-1]]
                cam = GradCAM(model=resnet_comp, target_layers=target_layers)
                t_cam = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor(), 
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                grayscale_cam = cam(input_tensor=t_cam(image).unsqueeze(0).to(DEVICE))[0, :]
                heatmap_vis = show_cam_on_image(np.array(image.resize((224,224)))/255.0, grayscale_cam, use_rgb=True)
                
                probs, boxes, labels = predict(image)
                img_nc2x = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    obj_name = COCO_CLASSES[labels[i]]
                    cv2.rectangle(img_nc2x, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_nc2x, obj_name, (x1, y1-10), 0, 0.7, (36, 255, 12), 2)

                c1, c2 = st.columns(2)
                with c1: st.image(heatmap_vis, caption="Grad-CAM (Visual Attention)", use_container_width=True)
                with c2: 
                    st.image(cv2.cvtColor(img_nc2x, cv2.COLOR_BGR2RGB), caption="NC2X (Graph Reasoning)", use_container_width=True)
                    render_knowledge_graph(labels)
