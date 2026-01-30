NC2X: Neuro-Symbolic Concept & Causal AI (SOTA V2)
NC2X is an advanced Explainable AI (XAI) framework that bridges the gap between traditional "Black Box" deep learning and human-understandable symbolic reasoning. Built on State-of-the-Art (SOTA) architectures and trained on high-compute hardware (NVIDIA RTX A6000), NC2X goes beyond simple classification by understanding the "Why" and "Context" of a scene.

Key Features
Neuro-Symbolic Reasoning: Combines deep visual features (Neural) with object relationships (Symbolic) using Graph Neural Networks (GNN).
SOTA Detection Engine: Powered by the latest YOLO11x (Extra Large) for unmatched object localization.
Deep Backbone: Uses ConvNeXt-V2 Large to extract 1536-dimensional global and local context vectors.
Causal Reasoning Engine: Allows users to perform "What-If" interventions by removing objects and observing logical confidence drops.
Dynamic Knowledge Graphs: Real-time generation of scene graphs showing how concepts relate to each other.
Comparative Analysis: Side-by-side comparison between Traditional Grad-CAM (Pixel-based) and NC2X (Graph-based) explanations.

Tech Stack & Architecture
1. Neural Backbone (The "Neuro" Part)
Model: convnextv2_large.fcmae_ft_in22k_in1k
Function: Extracts deep semantic features (1536-dim) to represent the visual essence of the scene.
2. Symbolic Extractor
Model: YOLO11x
Function: Identifies individual objects as "Symbols" to form nodes in our knowledge graph.
3. Reasoning Engine (The "Symbolic" Part)
Framework: PyTorch Geometric (PyG)
Architecture: Custom GNN with GCNConv layers.
Training: 100 Epochs on COCO 2017 Dataset.
4. Deployment
UI: Streamlit
Server: Hugging Face Spaces (Docker Environment)
Hardware: Optimized for NVIDIA RTX A6000.

Deploy Project Link: https://huggingface.co/spaces/jdwnx234/NC2X-App

Application Modes
🖼️ Image Analysis: Full scene breakdown with a dynamic Knowledge Graph.
🎥 Video Analysis: SOTA detection and symbolic tracking across video frames.
📸 Live Webcam: Interactive snapshot mode for real-time symbolic reasoning.
🧪 Causal Experiment: Manually mask objects to test the AI's contextual dependency.
⚖️ Comparison: Discover why Symbolic Graphs are superior to traditional Heatmaps.

🔧 Installation & Setup
If you want to run this project locally:
# Clone the repository
git clone https://github.com/suraj7880314386/nc2x-sota-app.git
cd nc2x-sota-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
Note: Ensure you have graphviz installed on your system.

📊 Causal Logic: The "Why"
Traditional AI might identify a "Dining Table" just by seeing chairs. NC2X tests this by removing the chairs:
Baseline: Table confidence = 99% (Chairs present).
Intervention: Chairs hidden.
Result: If confidence drops, NC2X proves a causal link between "Chairs" and "Table".
Developed as a State-of-the-Art Neuro-Symbolic Research Project.

📜 License
This project is licensed under the MIT License.
