import streamlit as st
import cv2
import numpy as np
import tempfile
import base64
import time
import os
import warnings
from collections import defaultdict

# Suppress all system and deep learning logs for absolute clean execution
warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Page configurations - clean layout with expanded settings sidebar
st.set_page_config(
    page_title="Retail / Mall Spatial Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark-Slate Corporate CSS Injection
st.markdown("""
<style>
    /* Clean Slate Core Colors */
    .stApp {
        background-color: #0d0f12;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Block */
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin-top: 1rem;
        margin-bottom: 0.1rem;
    }
    
    .subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* Metric Cards Overrides */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3b82f6 !important; /* Steel blue highlight */
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }



    /* Primary execution button styling */
    button[kind="primary"] {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border-radius: 4px !important;
        padding: 0.6rem 2.5rem !important;
        font-size: 1rem !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state for cached video files & properties
    if "cached_videos" not in st.session_state:
        st.session_state.cached_videos = {}

    def get_video_info(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": total_frames / fps
        }

    # Sidebar control for Processing Speed (Frame Skipping) only
    st.sidebar.markdown("### Settings")
    frame_skip = st.sidebar.slider(
        "Processing Speed (Frame Skipping)",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help="Set to 1 or 2 for maximum tracking accuracy. Set higher to process faster on lower-end systems."
    )
    
    device_option = st.sidebar.selectbox(
        "Target Hardware Device",
        options=["GPU (Hardware Accelerated)", "NPU (Intel AI Boost)", "CPU (Standard)"],
        index=0,
        help="Select the hardware device to accelerate the AI model. GPU dynamically uses CUDA (NVIDIA), MPS (Apple), or OpenVINO (Intel/AMD) depending on your system."
    )
    
    show_preview = st.sidebar.checkbox(
        "Show Live Video Preview",
        value=True,
        help="Uncheck this to disable live rendering. Recommended for very long videos to increase processing speed and avoid browser lag."
    )

    # Page Header
    st.markdown('<h1 class="main-title">Retail / Mall Spatial Analytics</h1>', unsafe_allow_html=True)

    # Video Input Selection
    st.markdown("### Video Input Selection")
    input_source = st.radio("Choose Input Method", ["Upload Video File", "Paste Video Link / Path"], horizontal=True, label_visibility="collapsed")
    
    video_paths = []
    video_names = []
    
    if input_source == "Upload Video File":
        uploaded_files = st.file_uploader(
            "Upload one or more video files to execute the vision pipeline",
            type=["mp4", "avi", "mov", "webm"],
            accept_multiple_files=True
        )
        if uploaded_files:
            current_keys = [f"{f.name}_{f.size}" for f in uploaded_files]
            
            # Clean up old temp files that are no longer selected
            existing_keys = list(st.session_state.cached_videos.keys())
            for key in existing_keys:
                if key not in current_keys:
                    try:
                        os.remove(st.session_state.cached_videos[key]["path"])
                    except:
                        pass
                    del st.session_state.cached_videos[key]
            
            # Process new uploads
            for f in uploaded_files:
                key = f"{f.name}_{f.size}"
                if key not in st.session_state.cached_videos:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(f.read())
                    tfile.close() # Close to allow cv2 to open
                    
                    info = get_video_info(tfile.name)
                    if info:
                        st.session_state.cached_videos[key] = {
                            "path": tfile.name,
                            "name": f.name,
                            "fps": info["fps"],
                            "total_frames": info["total_frames"],
                            "duration": info["duration"]
                        }
                    else:
                        try:
                            os.remove(tfile.name)
                        except:
                            pass
            
            for key in current_keys:
                if key in st.session_state.cached_videos:
                    video_paths.append(st.session_state.cached_videos[key]["path"])
                    video_names.append(st.session_state.cached_videos[key]["name"])
    else:
        video_urls_text = st.text_area(
            "Direct Video Links (URLs) or Local File Paths (One per line)", 
            placeholder="e.g.\nhttp://example.com/cctv_feed.mp4\nC:/CCTV/lobby.mp4"
        )
        if video_urls_text:
            lines = [line.strip().strip('"').strip("'") for line in video_urls_text.split("\n") if line.strip()]
            
            # Clean up old cached URL keys
            existing_keys = list(st.session_state.cached_videos.keys())
            for key in existing_keys:
                if key not in lines:
                    del st.session_state.cached_videos[key]
                    
            for key in lines:
                if key not in st.session_state.cached_videos:
                    info = get_video_info(key)
                    if info:
                        st.session_state.cached_videos[key] = {
                            "path": key,
                            "name": os.path.basename(key) if not key.startswith("http") else (key.split("/")[-1] or key),
                            "fps": info["fps"],
                            "total_frames": info["total_frames"],
                            "duration": info["duration"]
                        }
            
            for key in lines:
                if key in st.session_state.cached_videos:
                    video_paths.append(st.session_state.cached_videos[key]["path"])
                    video_names.append(st.session_state.cached_videos[key]["name"])

    # Render Timeline Range Sliders in the Sidebar for active video sources
    video_ranges = {}
    if st.session_state.cached_videos:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Video Analysis Timelines")
        
        for key, info in st.session_state.cached_videos.items():
            st.sidebar.markdown(f"**{info['name']}**")
            duration_secs = max(1.0, float(info["duration"]))
            duration_hours = float(duration_secs / 3600.0)
            
            selected_range = st.sidebar.slider(
                "Analysis Window (Hours)",
                min_value=0.0,
                max_value=duration_hours,
                value=(0.0, duration_hours),
                step=0.01,
                format="%.2f",
                key=f"slider_{key}"
            )
            
            def format_time_from_hours(h_float):
                s = h_float * 3600.0
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                sec = int(s % 60)
                return f"{h:02d}:{m:02d}:{sec:02d}"
                
            st.sidebar.caption(f"Selected Range: `{format_time_from_hours(selected_range[0])}` to `{format_time_from_hours(selected_range[1])}`")
            
            start_frame = int(selected_range[0] * 3600.0 * info["fps"])
            end_frame = int(selected_range[1] * 3600.0 * info["fps"])
            
            # Bound validation to ensure start is before end
            start_frame = max(0, min(start_frame, info["total_frames"] - 1))
            end_frame = max(start_frame + 1, min(end_frame, info["total_frames"]))
            
            video_ranges[info["name"]] = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_secs": selected_range[0],
                "end_secs": selected_range[1]
            }
    else:
        st.sidebar.markdown("---")
        st.sidebar.info("Upload a video or enter a file path to configure the analysis timeline window.")

    if video_paths:
        st.markdown("---")
        st.markdown("### Analytics Execution")
        
        # Start button
        if st.button("Execute", type="primary"):
            
            # Load YOLO Small model silently (highly robust for distant tracking)
            with st.spinner("Initializing hardware-accelerated neural networks..."):
                try:
                    import torch
                    from ultralytics import YOLO
                    
                    # Detect available hardware components
                    has_cuda = torch.cuda.is_available()
                    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                    
                    has_ov_gpu = False
                    has_ov_npu = False
                    try:
                        from openvino import Core
                        core = Core()
                        ov_devices = core.available_devices
                        if "GPU" in ov_devices:
                            has_ov_gpu = True
                        if "NPU" in ov_devices:
                            has_ov_npu = True
                    except:
                        pass

                    device_str = "cpu"
                    use_openvino = False
                    detected_device_name = "CPU"

                    if device_option == "GPU (Hardware Accelerated)":
                        if has_cuda:
                            device_str = "cuda"
                            detected_device_name = "NVIDIA GPU (CUDA) via PyTorch"
                        elif has_mps:
                            device_str = "mps"
                            detected_device_name = "Apple Silicon GPU (MPS) via PyTorch"
                        elif has_ov_gpu:
                            device_str = "intel:gpu"
                            use_openvino = True
                            detected_device_name = "Intel/AMD GPU (OpenVINO)"
                        else:
                            # Fallback if no specific accelerator detected
                            device_str = "cpu"
                            detected_device_name = "CPU (No GPU acceleration available)"
                    elif device_option == "NPU (Intel AI Boost)":
                        if has_ov_npu:
                            device_str = "intel:npu"
                            use_openvino = True
                            detected_device_name = "Intel AI Boost NPU"
                        else:
                            st.warning("Intel AI Boost NPU was selected but no NPU device was detected. Falling back to CPU...")
                            device_str = "cpu"
                            detected_device_name = "CPU (Fallback)"
                    else:
                        device_str = "cpu"
                        detected_device_name = "CPU (Standard)"

                    st.info(f"Targeting execution device: **{detected_device_name}**")

                    if use_openvino:
                        model_name = "yolov8s_openvino_model"
                        # Check if optimized OpenVINO model exists, if not export it
                        if not os.path.exists(model_name):
                            st.info("First-time setup: Exporting YOLO model to OpenVINO format for hardware acceleration. This takes about 30 seconds...")
                            base_model = YOLO("yolov8s.pt")
                            base_model.export(format="openvino")
                            
                        # Load OpenVINO model
                        model = YOLO(model_name)
                    else:
                        model = YOLO("yolov8s.pt")
                except Exception as e:
                    st.warning(f"Failed to initialize optimized hardware device ({device_option}). Error: {e}")
                    st.info("Falling back to standard CPU processing...")
                    try:
                        from ultralytics import YOLO
                        model = YOLO("yolov8s.pt")
                        device_str = "cpu"
                    except Exception as fallback_err:
                        st.error(f"Failed to load fallback AI model: {fallback_err}")
                        return

            all_results = []

            # Streamlit UI placeholders that will be updated for each video inside a clearable container
            processing_placeholder = st.empty()
            with processing_placeholder.container():
                st.markdown("### Processing Active Feeds")
                col_video, col_stats = st.columns([7, 3])

                with col_video:
                    active_video_title = st.empty()
                    video_placeholder = st.empty()
                    progress_bar = st.progress(0.0)

                with col_stats:
                    st.markdown("#### Frame-Level Statistics")
                    metric_total = st.empty()
                    metric_active = st.empty()
                    metric_total.metric("Total People Tracked", "0")
                    metric_active.metric("Current Shoppers in View", "0")
                    status_placeholder = st.empty()
                    status_placeholder.info("Processing initialized...")

            # Run through each video source sequentially
            for video_idx, (video_path, video_name) in enumerate(zip(video_paths, video_names)):
                active_video_title.markdown(f"#### Processing Camera {video_idx + 1}/{len(video_paths)}: `{video_name}`")
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error(f"Error opening video: {video_name}")
                    continue

                # Read first frame to initialize spatial matrix
                ret, first_frame = cap.read()
                if not ret:
                    st.error(f"Failed to decode video frames for: {video_name}")
                    cap.release()
                    continue

                orig_height, orig_width, _ = first_frame.shape
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 30.0
                
                # Hardcoded Peak Precision Parameters (Zero Configuration)
                proc_width = 1280
                scale_ratio = proc_width / float(orig_width)
                proc_height = int(orig_height * scale_ratio)
                
                # Processing settings
                conf_threshold = 0.15 # Capture tiny distant shoppers
                # frame_skip is dynamically assigned from user sidebar input above

                # Initialize unique shopper grid mapping for volume-based crowd overlap (Standard Human Definition)
                grid_scale = 8
                grid_w = max(1, orig_width // grid_scale)
                grid_h = max(1, orig_height // grid_scale)
                # Maps (grid_y, grid_x) -> set of unique Track IDs that visited this cell
                visitor_cells = defaultdict(set)
                
                unique_visitors = set()
                path_history = {}
                sector_detections = defaultdict(int)
                reference_frame = first_frame.copy()

                # Dynamic color generation
                def get_color(track_id):
                    np.random.seed(int(track_id))
                    return [int(c) for c in np.random.randint(50, 255, size=3)]

                # Retrieve timeline bounds if configured
                start_frame = 0
                end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if video_name in video_ranges:
                    start_frame = video_ranges[video_name]["start_frame"]
                    end_frame = video_ranges[video_name]["end_frame"]
                
                # Fetch a frame at start_frame to use as the background reference frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret_ref, ref_frame_temp = cap.read()
                if ret_ref:
                    reference_frame = ref_frame_temp.copy()
                
                # Reset capture to start_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_count = start_frame
                processed_frame_count = 0
                last_preview_time = 0.0
                frames_to_process = end_frame - start_frame

                while cap.isOpened():
                    if frame_count >= end_frame:
                        break
                    
                    # Perfect skip interval relative to start_frame
                    if (frame_count - start_frame) % frame_skip != 0:
                        ret = cap.grab()
                        frame_count += 1
                        if not ret:
                            break
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    processed_frame_count += 1
                    current_active_ids = set()

                    # Scale frame to 1280px details for distant accuracy
                    small_frame = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA)

                    # Process strictly Person class (class 0)
                    tracker_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_tracker.yaml")
                    if not os.path.exists(tracker_config):
                        tracker_config = "bytetrack.yaml"

                    # Determine tracking device parameter for YOLO
                    if 'device_str' in locals():
                        track_device = device_str
                    else:
                        track_device = "cpu"

                    results = model.track(
                        source=small_frame,
                        persist=True,
                        classes=[0],
                        conf=conf_threshold,
                        tracker=tracker_config,
                        verbose=False,
                        device=track_device
                    )

                    if results and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None

                        for idx, box in enumerate(boxes):
                            # Convert back to high-res original canvas coordinates
                            x1 = int(box[0] / scale_ratio)
                            y1 = int(box[1] / scale_ratio)
                            x2 = int(box[2] / scale_ratio)
                            y2 = int(box[3] / scale_ratio)

                            # Floor/foot coordinates (center of bounding box bottom)
                            foot_x = int((x1 + x2) / 2)
                            foot_y = y2

                            # Log visitor count into spatial sector grid (cumulative presence)
                            sr = min(5, int(foot_y / (orig_height / 6.0)))
                            sc = min(5, int(foot_x / (orig_width / 6.0)))
                            sector_detections[(sr, sc)] += 1

                            # If tracked, perform tracker-specific operations
                            if track_ids is not None and idx < len(track_ids):
                                track_id = int(track_ids[idx])
                                current_active_ids.add(track_id)
                                unique_visitors.add(track_id)

                                # Store trail path history
                                if track_id not in path_history:
                                    path_history[track_id] = []
                                path_history[track_id].append((foot_x, foot_y))
                                if len(path_history[track_id]) > 25:
                                    path_history[track_id].pop(0)

                                # Log track ID into unique spatial cells once every 5 processed frames to optimize CPU overhead
                                if processed_frame_count % 5 == 0:
                                    cx = foot_x // grid_scale
                                    cy = foot_y // grid_scale
                                    for dy in range(-3, 4):
                                        for dx in range(-3, 4):
                                            ny, nx = cy + dy, cx + dx
                                            if 0 <= ny < grid_h and 0 <= nx < grid_w:
                                                visitor_cells[(ny, nx)].add(track_id)

                                # Draw trails
                                trail = path_history[track_id]
                                for i in range(1, len(trail)):
                                    cv2.line(frame, trail[i-1], trail[i], get_color(track_id), 2, cv2.LINE_AA)

                                # Draw elegant bounding box
                                color = get_color(track_id)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Minimal ID label
                                label = f"ID: {track_id}"
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                                cv2.rectangle(frame, (x1, y1 - 15), (x1 + w, y1), color, -1)
                                cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                            else:
                                # Untracked person detected precisely - draw a thin grey box to show detection
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

                    # Minimal HUD Overlay
                    hud = frame.copy()
                    cv2.rectangle(hud, (10, 10), (320, 100), (12, 16, 23), -1)
                    cv2.addWeighted(hud, 0.8, frame, 0.2, 0, frame)
                    
                    # Active Status bar
                    cv2.circle(frame, (25, 30), 5, (59, 130, 246), -1)
                    cv2.putText(frame, "CCTV LIVE VISION INTERFACE", (40, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"TOTAL COUNTED: {len(unique_visitors)}", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"CURRENT IN VIEW: {len(current_active_ids)}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

                    # ponytail: preview capped at ~4 fps, 960px, JPEG q60 (~60 KB) so it fits a slow link; raise if bandwidth allows
                    if show_preview and time.time() - last_preview_time >= 0.25:
                        last_preview_time = time.time()
                        # Send the frame inline as a JPEG data URI. A plain st.image(array) makes the browser
                        # download each frame separately; from a remote server the next frame cancels that
                        # download before it finishes, so the preview never updates (only works on localhost).
                        preview = frame if orig_width <= 960 else cv2.resize(frame, (960, int(orig_height * 960 / orig_width)), interpolation=cv2.INTER_AREA)
                        _, jpg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        video_placeholder.markdown(f'<img src="data:image/jpeg;base64,{base64.b64encode(jpg).decode()}" style="width:100%;border-radius:4px">', unsafe_allow_html=True)

                        # Update KPI metrics
                        metric_total.metric("Total People Tracked", str(len(unique_visitors)))
                        metric_active.metric("Current Shoppers in View", str(len(current_active_ids)))

                        # Update progress bar
                        if frames_to_process > 0:
                            pct = float(frame_count - start_frame) / float(frames_to_process)
                            progress_bar.progress(min(pct, 1.0))
                            status_placeholder.info(f"Analyzing: {video_name} (Frame {frame_count}/{end_frame})")
                    else:
                        # Update text and progress bar less frequently to optimize processing speed and avoid WebSocket issues
                        if processed_frame_count % 30 == 0 or frame_count >= end_frame:
                            # Update KPI metrics
                            metric_total.metric("Total People Tracked", str(len(unique_visitors)))
                            metric_active.metric("Current Shoppers in View", str(len(current_active_ids)))

                            # Update progress bar
                            if frames_to_process > 0:
                                pct = float(frame_count - start_frame) / float(frames_to_process)
                                progress_bar.progress(min(pct, 1.0))
                                status_placeholder.info(f"Analyzing: {video_name} (Frame {frame_count}/{end_frame})")

                cap.release()
                status_placeholder.info(f"Vision analysis complete for {video_name}. Compiling heatmap...")

                # ---------------------------------------------
                # CINEMATIC GAUSSIAN HEATMAP GENERATION
                # ---------------------------------------------
                
                # Compile unique visitor overlap grid
                overlap_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
                for (cy, cx), visitors in visitor_cells.items():
                    overlap_grid[cy, cx] = len(visitors)

                # Resize the unique count grid back to original high-resolution size
                heatmap_accum = cv2.resize(overlap_grid, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

                # Normalize matrix
                max_accum = np.max(heatmap_accum)
                if max_accum > 0:
                    heatmap_norm = np.clip(heatmap_accum * (255.0 / max_accum), 0, 255).astype(np.uint8)
                else:
                    heatmap_norm = heatmap_accum.astype(np.uint8)

                # Smooth Gaussian blur (101x101) for fluid thermal glow
                heatmap_blur = cv2.GaussianBlur(heatmap_norm, (101, 101), 0)
                heatmap_color = cv2.applyColorMap(heatmap_blur, cv2.COLORMAP_JET)

                # Limit boundary mask (at least 4% of peak density)
                mask = heatmap_blur > (np.max(heatmap_blur) * 0.04)
                
                # Preserve original background brightness (95%)
                dimmed_frame = cv2.convertScaleAbs(reference_frame, alpha=0.95, beta=0)
                blended_heatmap = dimmed_frame.copy()
                
                if np.max(heatmap_blur) > 0:
                    # Highly transparent blend (75% original background, 25% subtle heatmap)
                    blended_heatmap[mask] = cv2.addWeighted(dimmed_frame, 0.75, heatmap_color, 0.25, 0)[mask]

                # Mark the single busiest spot on the heatmap
                if np.max(heatmap_blur) > 0:
                    _, _, _, (px, py) = cv2.minMaxLoc(heatmap_blur)
                    peak_scale = max(0.5, (orig_width / 1920.0) * 0.7)
                    peak_thick = max(1, int(orig_width / 1920.0 * 2))
                    radius = max(25, int(orig_width / 1920.0 * 45))
                    cv2.circle(blended_heatmap, (px, py), radius, (0, 0, 255), peak_thick, cv2.LINE_AA)
                    cv2.circle(blended_heatmap, (px, py), max(3, radius // 10), (0, 0, 255), -1)
                    label_text = "PEAK CONGESTION"
                    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, peak_scale, peak_thick)
                    # Keep the label inside the frame
                    tx = min(max(px - w // 2, 8), orig_width - w - 8)
                    ty = max(py - max(30, int(orig_height * 0.04)), h + 8)
                    cv2.rectangle(blended_heatmap, (tx - 8, ty - h - 6), (tx + w + 8, ty + 6), (0, 0, 255), -1)
                    cv2.putText(blended_heatmap, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, peak_scale, (255, 255, 255), peak_thick, cv2.LINE_AA)

                # ---------------------------------------------
                # SPATIAL GRID CLASSIFICATION ENGINE
                # ---------------------------------------------
                blended_flow_map = reference_frame.copy()
                
                # Find maximum detection count in any sector
                max_sector_detections = 0
                for sr in range(6):
                    for sc in range(6):
                        count = sector_detections[(sr, sc)]
                        if count > max_sector_detections:
                            max_sector_detections = count
                
                dominant_count = 0
                dead_count = 0
                underutilized_count = 0
                
                font_scale = max(0.4, (orig_width / 1920.0) * 0.55)
                thickness = max(1, int(orig_width / 1920.0 * 2))
                
                for sr in range(6):
                    for sc in range(6):
                        count = sector_detections[(sr, sc)]
                        ratio = count / max_sector_detections if max_sector_detections > 0 else 0
                        
                        # Sector pixel coordinates
                        x1_sec = int(sc * (orig_width / 6.0))
                        y1_sec = int(sr * (orig_height / 6.0))
                        x2_sec = int((sc + 1) * (orig_width / 6.0))
                        y2_sec = int((sr + 1) * (orig_height / 6.0))
                        
                        color = None
                        label = None
                        
                        if ratio >= 0.20:
                            dominant_count += 1
                            color = (0, 200, 0) # Green
                            label = "DOMINANT"
                        elif count == 0:
                            dead_count += 1
                            color = (0, 0, 220) # Red
                            label = "DEAD AREA"
                        else:
                            underutilized_count += 1
                            color = (200, 50, 0) # Blue
                            label = "UNDERUTILIZED"
                        
                        if color is not None:
                            # Render sector overlay fill (15% transparency)
                            overlay = blended_flow_map[y1_sec:y2_sec, x1_sec:x2_sec]
                            color_mask = np.full(overlay.shape, color, dtype=np.uint8)
                            blended_flow_map[y1_sec:y2_sec, x1_sec:x2_sec] = cv2.addWeighted(overlay, 0.85, color_mask, 0.15, 0)
                            
                            # Draw solid border
                            cv2.rectangle(blended_flow_map, (x1_sec, y1_sec), (x2_sec, y2_sec), color, 2)
                            
                            # Draw label badge
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                            cv2.rectangle(blended_flow_map, (x1_sec + 2, y1_sec + 2), (x1_sec + tw + 10, y1_sec + th + 10), color, -1)
                            cv2.putText(blended_flow_map, label, (x1_sec + 7, y1_sec + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


                # Store result
                all_results.append({
                    "video_name": video_name,
                    "video_path": video_path,
                    "unique_visitors_count": len(unique_visitors),
                    "blended_heatmap": blended_heatmap,
                    "blended_flow_map": blended_flow_map,
                    "pct_dominant": (dominant_count / 36.0) * 100.0,
                    "pct_dead": (dead_count / 36.0) * 100.0,
                    "pct_underutilized": (underutilized_count / 36.0) * 100.0,
                })

            # Clear active processing panel entirely to free up screen space
            processing_placeholder.empty()

            if not all_results:
                st.error("No video feeds were successfully processed. Please verify that your Google Drive is mounted in Colab and that the file path is correct.")
                return

            # ---------------------------------------------
            # METRICS & TABBED VISUALIZATION
            # ---------------------------------------------
            st.markdown("---")
            st.markdown("### Spatial Heatmaps & Congestion Reports")

            # Display Camera Results sequentially (One after another)
            for idx, r in enumerate(all_results):
                st.markdown(f"## Camera {idx+1}: {r['video_name']}")
                
                # 1. Spatial Traffic Heatmap Section
                st.markdown("### 1. Spatial Traffic Heatmap")
                col_map, col_details = st.columns([6, 4])
                with col_map:
                    blended_rgb = cv2.cvtColor(r["blended_heatmap"], cv2.COLOR_BGR2RGB)
                    st.image(blended_rgb, caption=f"Spatial movement heatmap for {r['video_name']}", width="stretch")
                with col_details:
                    st.markdown("#### Spatial Metrics Summary")
                    st.markdown(f"**Total Pedestrians Verified**: `{r['unique_visitors_count']}` unique tracks audited.")

                st.markdown("---")
                
                # 2. Floor Space Classification Section
                st.markdown("### 2. Floor Space Classification")
                col_flow, col_flow_details = st.columns([6, 4])
                with col_flow:
                    flow_rgb = cv2.cvtColor(r["blended_flow_map"], cv2.COLOR_BGR2RGB)
                    st.image(flow_rgb, caption=f"Spatial Zone Classification overlay for {r['video_name']}", width="stretch")
                    
                with col_flow_details:
                    st.markdown("#### Floor Space Classification Metrics")
                    st.markdown(f"* **Dominant Paths (High Traffic)**: `{r['pct_dominant']:.1f}%` of floor space.")
                    st.markdown(f"* **Underutilized Zones (Low Traffic)**: `{r['pct_underutilized']:.1f}%` of floor space.")
                    st.markdown(f"* **Dead Areas (Zero Traffic)**: `{r['pct_dead']:.1f}%` of floor space.")
                    
                    st.markdown(f"""
                    <div style="margin-top: 25px;">
                        <strong>Visual Floor Space Breakdown:</strong>
                        <div style="display: flex; height: 20px; border-radius: 4px; overflow: hidden; margin-top: 8px; border: 1px solid #222b3c;">
                            <div style="width: {r['pct_dominant']}%; background-color: #22c55e;" title="Dominant Paths: {r['pct_dominant']:.1f}%"></div>
                            <div style="width: {r['pct_underutilized']}%; background-color: #3b82f6;" title="Underutilized Zones: {r['pct_underutilized']:.1f}%"></div>
                            <div style="width: {r['pct_dead']}%; background-color: #ef4444;" title="Dead Areas: {r['pct_dead']:.1f}%"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
                            <span>Dominant (Green)</span>
                            <span>Underutilized (Blue)</span>
                            <span>Dead (Red)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add horizontal separator between different cameras
                st.markdown("<hr style='border: 2px solid #222b3c; margin-top: 3rem; margin-bottom: 3rem;'>", unsafe_allow_html=True)



if __name__ == "__main__":
    import streamlit as st
    if st.runtime.exists():
        main()
    else:
        import sys
        from streamlit.web import cli as stcli
        # Set max upload size (in MB) directly here
        max_upload_mb = "2048"
        sys.argv = ["streamlit", "run", __file__, "--server.maxUploadSize", max_upload_mb] + sys.argv[1:]
        sys.exit(stcli.main())
