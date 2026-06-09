# Retail & Mall Spatial Analytics Dashboard

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-005A9C.svg)](https://docs.openvino.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Transform existing CCTV infrastructure into a physical-world business intelligence and spatial memory system. Instead of utilizing surveillance footage purely for manual security reviews, this system continuously processes video feeds to quantify human behavior, spatial layout efficiency, and customer interaction dynamics.

---

## 🎯 Project Vision & Business Intelligence

In retail environments, understanding spatial utilization is the key to maximizing revenue. This project turns passive CCTV camera video feeds into actionable insights:
* **Foot Traffic Measurement**: Count cumulative unique shoppers and monitor real-time occupant presence.
* **Spatial Optimization**: Categorize physical zones into high-traffic (Dominant), underutilized, or dead space to optimize mall rents and store layouts.
* **Visual Attention Mapping**: Trace shopper orientation and project eye-level gaze attention areas to evaluate storefront and advertisement visibility.
* **Operational Intelligence**: Enable operators to detect bottlenecks, optimize floor layouts, and place high-value ads in scientifically proven hotspots.

---

## 🚀 Key Features

### 1. High-Fidelity Shopper Tracking
* Powered by **YOLOv8** Object Detection coupled with a customized **ByteTrack** tracking algorithm.
* Optimized tracking sensitivity via [custom_tracker.yaml](file:///c:/Users/POOJA/Documents/vissioning/custom_tracker.yaml) for reliable long-range tracking, even in crowded or low-resolution atrium environments.

### 2. Multi-Dimensional Visual Analytics
* **Movement Heatmap (`heatmap_*.png`)**: Generates smooth Gaussian-blurred density clouds overlaid on the camera's reference frame to highlight dwell patterns and peak traffic regions.
* **Spatial Zone Classification (`flow_*.png`)**: Segments the area into a 6x6 spatial grid, classifying zones dynamically:
  * 🟩 **Dominant Paths**: Extremely high traffic and movement pathways.
  * 🟦 **Underutilized Zones**: Light foot traffic with potential for optimization.
  * 🟥 **Dead Areas**: Zero foot traffic recorded during the analysis window.
* **Visual Attention & Gaze Hotspots (`attention_*.png`)**: Projects gaze lines based on tracking velocity. Fast-moving shoppers project attention forward along their path, while stationary/dwelling shoppers project a circular attention field. Locates and ranks the Top 3 attention peaks.
* **Annotated Video Output (`output_*.mp4`)**: Exports full-resolution video showing tracking boxes, active shopper tags, movement trails, custom zone overlays, and a HUD status display.

### 3. Dynamic Zone Segmentation
* **Vertical Segmentation**: Splits the scene into Left, Center, and Right zones to compare storefront entry paths.
* **Horizontal Segmentation (Multi-Floor View)**: Auto-detects physical floor walkways/balconies in shopping mall atriums using shopper density projection histograms, tracking multi-floor counts simultaneously.

---

## 💻 Hardware Acceleration Engine

The vision pipeline features automatic hardware discovery. It searches and compiles optimized models based on your systems:
1. **GPU (NVIDIA CUDA / Apple Silicon MPS)**: Accelerates inference via PyTorch hardware backends.
2. **NPU (Intel AI Boost)**: Compiles and executes the model utilizing Intel OpenVINO.
3. **Intel/AMD GPU (OpenVINO)**: Exports YOLOv8 weights into OpenVINO Intermediate Representation (`IR`) for sub-millisecond execution.
4. **CPU (Standard)**: Automatically falls back to standard CPU thread processing if no dedicated accelerators are present.

---

## 📂 Project Structure

```
├── app.py                # Premium Streamlit Multi-Video Dashboard Web Application
├── pipeline.py           # Command-Line Batch Processing Script
├── custom_tracker.yaml   # Configured ByteTrack thresholds for high-sensitivity tracking
├── requirements.txt      # Python package dependencies
├── .gitignore            # Git exclusion list (excludes models, caches, and output media)
└── readme.md             # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/poojamaskare/Retail-Crowd-Analytics.git
cd Retail-Crowd-Analytics
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 How to Run the Applications

The project includes two interfaces: an interactive web dashboard and a command-line script.

### Option A: Interactive Streamlit Dashboard (Recommended)

Start the premium, dark-themed dashboard which supports multi-video uploads, live video previews, frame skipping controls, real-time KPI card updates, and interactive timeline cropping.

```bash
streamlit run app.py
```

* **Live Dashboard Controls**:
  * **Processing Speed (Frame Skipping)**: Balance tracking resolution with performance.
  * **Zone Division Mode**: Switch between Vertical, Horizontal (Multi-Floor), and Disabled.
  * **Target Hardware Device**: Choose GPU, Intel NPU, or CPU.
  * **Video Timeline Ranges**: Crop analysis start and end times dynamically inside the sidebar.

---

### Option B: Command-Line Processing Script

For batch processing, cron automation, or server runs, use `pipeline.py`.

```bash
# Basic running with default settings on a video file
python pipeline.py --input path/to/cctv_footage.mp4

# Run with customized confidence thresholds, frame skipping, and floor/horizontal zones
python pipeline.py --input cctv_1.mp4,cctv_2.mp4 --confidence 0.25 --skip 2 --zones horizontal --device gpp