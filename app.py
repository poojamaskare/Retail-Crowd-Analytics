import streamlit as st
import cv2
import numpy as np
import tempfile
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
    
    zone_mode = st.sidebar.selectbox(
        "Zone Division Mode",
        options=["Vertical (Left, Center, Right)", "Horizontal (Multi-Floor View)", "Disabled"],
        index=0,
        help="Choose how to segment the spatial perspective into tracking zones or floors."
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

    # Default floor variables (will be dynamically detected from shopper coordinates)
    num_floors = 3
    floor_boundaries = [0.20, 0.50]

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

            # Helper: Draw Dashed Line
            def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=15):
                dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                pts = []
                for i in np.arange(0, dist, gap):
                    r = i / dist
                    pts.append((int((1-r)*pt1[0] + r*pt2[0]), int((1-r)*pt1[1] + r*pt2[1])))
                for i in range(0, len(pts)-1, 2):
                    cv2.line(img, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)

            # Helper: Draw Zone Label
            def draw_zone_label(img, text, pt, color, font_scale=0.5, thickness=1):
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(img, (pt[0] - 5, pt[1] - h - 5), (pt[0] + w + 5, pt[1] + 5), (12, 16, 23), -1)
                cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

            # Helper: Detect Atrium Floors Automatically using vertical shopper density projection
            def detect_atrium_floors(y_coords, height):
                if len(y_coords) < 30:
                    return [0.20, 0.50], ["Second Floor", "First Floor", "Ground Floor"]
                
                # Compute histogram of Y coordinates (vertical density)
                bins = 40
                hist, bin_edges = np.histogram(y_coords, bins=bins, range=(0, height))
                
                # Smooth the histogram to eliminate tracking noise
                hist_smooth = np.convolve(hist, np.ones(5)/5.0, mode='same')
                
                # Find local density peaks corresponding to physical floor walkways
                peaks = []
                for i in range(1, len(hist_smooth) - 1):
                    if hist_smooth[i] >= hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                        density = hist_smooth[i]
                        y_val = (bin_edges[i] + bin_edges[i+1]) / 2.0
                        if density > (len(y_coords) * 0.02):  # filter out noise peaks
                            peaks.append((density, y_val))
                
                peaks = sorted(peaks, key=lambda x: x[0], reverse=True)
                
                # Filter peaks that are separated by at least 15% of frame height (physically different levels)
                filtered_peaks = []
                for density, y_val in peaks:
                    too_close = False
                    for _, fy in filtered_peaks:
                        if abs(fy - y_val) < (height * 0.15):
                            too_close = True
                            break
                    if not too_close:
                        filtered_peaks.append((density, y_val))
                
                selected_peaks = sorted([fy for _, fy in filtered_peaks])
                num_peaks = len(selected_peaks)
                
                if num_peaks <= 1:
                    # Single floor detected
                    return [], ["Ground Floor"]
                elif num_peaks == 2:
                    # Two floors detected
                    split = (selected_peaks[0] + selected_peaks[1]) / 2.0 / height
                    return [split], ["First Floor", "Ground Floor"]
                else:
                    # Three or more floors detected (take the top 3 densest peaks)
                    selected_peaks = selected_peaks[:3]
                    selected_peaks = sorted(selected_peaks)
                    if len(selected_peaks) == 3:
                        split1 = (selected_peaks[0] + selected_peaks[1]) / 2.0 / height
                        split2 = (selected_peaks[1] + selected_peaks[2]) / 2.0 / height
                        return [split1, split2], ["Second Floor", "First Floor", "Ground Floor"]
                    elif len(selected_peaks) == 2:
                        split = (selected_peaks[0] + selected_peaks[1]) / 2.0 / height
                        return [split], ["First Floor", "Ground Floor"]
                    else:
                        return [], ["Ground Floor"]

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
                
                # Visual Attention Accumulator Grid
                attention_accum = np.zeros((grid_h, grid_w), dtype=np.float32)
                
                unique_visitors = set()
                path_history = {}
                sector_detections = defaultdict(int)
                all_detected_y_coords = []
                reference_frame = first_frame.copy()

                # Initialize Zone Occupancy Counts (Cumulative occupant presence)
                zone_occupancy = {}
                floors = []
                if num_floors == 1:
                    floors = ["Ground Floor"]
                elif num_floors == 2:
                    floors = ["First Floor", "Ground Floor"]
                else:
                    floors = ["Second Floor", "First Floor", "Ground Floor"]

                if "Vertical" in zone_mode:
                    zone_occupancy = {"Left Zone": 0, "Center Zone": 0, "Right Zone": 0}
                elif "Horizontal" in zone_mode:
                    zone_occupancy = {f: 0 for f in floors}

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
                            all_detected_y_coords.append(foot_y)

                            # Increment zone/floor occupancy counters
                            if "Vertical" in zone_mode:
                                if foot_x < orig_width // 3:
                                    zone_occupancy["Left Zone"] += 1
                                elif foot_x < 2 * orig_width // 3:
                                    zone_occupancy["Center Zone"] += 1
                                else:
                                    zone_occupancy["Right Zone"] += 1
                            elif "Horizontal" in zone_mode:
                                if num_floors == 1:
                                    zone_occupancy["Ground Floor"] += 1
                                elif num_floors == 2:
                                    if foot_y < int(orig_height * floor_boundaries[0]):
                                        zone_occupancy["First Floor"] += 1
                                    else:
                                        zone_occupancy["Ground Floor"] += 1
                                else:
                                    if foot_y < int(orig_height * floor_boundaries[0]):
                                        zone_occupancy["Second Floor"] += 1
                                    elif foot_y < int(orig_height * floor_boundaries[1]):
                                        zone_occupancy["First Floor"] += 1
                                    else:
                                        zone_occupancy["Ground Floor"] += 1

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

                                # Compute movement direction vector for gaze projection
                                hx = foot_x
                                hy = y1
                                
                                history = path_history[track_id]
                                if len(history) >= 3:
                                    dx = foot_x - history[-3][0]
                                    dy = foot_y - history[-3][1]
                                    speed = np.sqrt(dx**2 + dy**2)
                                else:
                                    speed = 0
                                    
                                if speed > 5:
                                    # Person is walking: project a gaze line forward in the direction of movement
                                    vx = dx / speed
                                    vy = dy / speed
                                    
                                    # Gaze line: from head_x, head_y forward by 160 pixels
                                    # Sample 8 points along the gaze line to update the attention accumulator
                                    for step in range(1, 9):
                                        sample_x = hx + vx * (step * 20)
                                        sample_y = hy + vy * (step * 20)
                                        ngx = int(sample_x / grid_scale)
                                        ngy = int(sample_y / grid_scale)
                                        if 0 <= ngy < grid_h and 0 <= ngx < grid_w:
                                            # Decay weight slightly as distance from observer increases
                                            weight = 1.5 - (step * 0.1)
                                            attention_accum[ngy, ngx] += weight
                                else:
                                    # Person is standing/dwelling: project a circular scan field around their position
                                    for dy_px in range(-30, 31, 10):
                                        for dx_px in range(-30, 31, 10):
                                            dist = np.sqrt(dx_px**2 + dy_px**2)
                                            if dist <= 30:
                                                ngx = int((hx + dx_px) / grid_scale)
                                                ngy = int((hy + dy_px) / grid_scale)
                                                if 0 <= ngy < grid_h and 0 <= ngx < grid_w:
                                                    attention_accum[ngy, ngx] += 1.0

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

                    # Draw Zone Separator Lines and Labels on Live Frame
                    font_scale = max(0.5, (orig_width / 1920.0) * 0.7)
                    thickness = max(1, int(orig_width / 1920.0 * 2))
                    
                    if "Vertical" in zone_mode:
                        draw_dashed_line(frame, (orig_width // 3, 0), (orig_width // 3, orig_height), (0, 165, 255), thickness)
                        draw_dashed_line(frame, (2 * orig_width // 3, 0), (2 * orig_width // 3, orig_height), (0, 165, 255), thickness)
                        draw_zone_label(frame, f"LEFT ZONE: {zone_occupancy['Left Zone']}", (orig_width // 6 - 50, 40), (0, 165, 255), font_scale, thickness)
                        draw_zone_label(frame, f"CENTER ZONE: {zone_occupancy['Center Zone']}", (orig_width // 2 - 60, 40), (0, 165, 255), font_scale, thickness)
                        draw_zone_label(frame, f"RIGHT ZONE: {zone_occupancy['Right Zone']}", (5 * orig_width // 6 - 60, 40), (0, 165, 255), font_scale, thickness)
                    elif "Horizontal" in zone_mode:
                        # Draw floor labels directly on the live frame without grid divider lines
                        if num_floors == 1:
                            draw_zone_label(frame, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(orig_height * 0.50)), (255, 165, 0), font_scale, thickness)
                        elif num_floors == 2:
                            draw_zone_label(frame, f"FIRST FLOOR: {zone_occupancy['First Floor']}", (20, int(orig_height * (floor_boundaries[0] / 2))), (255, 165, 0), font_scale, thickness)
                            draw_zone_label(frame, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(orig_height * ((floor_boundaries[0] + 1.0) / 2))), (255, 165, 0), font_scale, thickness)
                        else:
                            draw_zone_label(frame, f"SECOND FLOOR: {zone_occupancy['Second Floor']}", (20, int(orig_height * (floor_boundaries[0] / 2))), (255, 165, 0), font_scale, thickness)
                            draw_zone_label(frame, f"FIRST FLOOR: {zone_occupancy['First Floor']}", (20, int(orig_height * ((floor_boundaries[0] + floor_boundaries[1]) / 2))), (255, 165, 0), font_scale, thickness)
                            draw_zone_label(frame, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(orig_height * ((floor_boundaries[1] + 1.0) / 2))), (255, 165, 0), font_scale, thickness)

                    # Minimal HUD Overlay
                    hud = frame.copy()
                    cv2.rectangle(hud, (10, 10), (320, 100), (12, 16, 23), -1)
                    cv2.addWeighted(hud, 0.8, frame, 0.2, 0, frame)
                    
                    # Active Status bar
                    cv2.circle(frame, (25, 30), 5, (59, 130, 246), -1)
                    cv2.putText(frame, "CCTV LIVE VISION INTERFACE", (40, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"TOTAL COUNTED: {len(unique_visitors)}", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"CURRENT IN VIEW: {len(current_active_ids)}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

                    if show_preview:
                        # Convert to RGB for Streamlit rendering
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

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

                # Automatically detect atrium floor levels from accumulated shopper vertical density
                if "Horizontal" in zone_mode:
                    floor_boundaries, floors = detect_atrium_floors(all_detected_y_coords, orig_height)
                    num_floors = len(floors)
                    
                    # Recalculate zone_occupancy with the exact auto-detected splits
                    zone_occupancy = {f: 0 for f in floors}
                    for y in all_detected_y_coords:
                        if len(floor_boundaries) == 0:
                            zone_occupancy["Ground Floor"] += 1
                        elif len(floor_boundaries) == 1:
                            if y < int(orig_height * floor_boundaries[0]):
                                zone_occupancy["First Floor"] += 1
                            else:
                                zone_occupancy["Ground Floor"] += 1
                        else:
                            if y < int(orig_height * floor_boundaries[0]):
                                zone_occupancy["Second Floor"] += 1
                            elif y < int(orig_height * floor_boundaries[1]):
                                zone_occupancy["First Floor"] += 1
                            else:
                                zone_occupancy["Ground Floor"] += 1

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

                # Determine zone regions for separate peak checking
                zone_hotspots = []
                
                if "Vertical" in zone_mode:
                    regions = [
                        ("Left Zone", (0, 0, orig_width // 3, orig_height)),
                        ("Center Zone", (orig_width // 3, 0, 2 * orig_width // 3, orig_height)),
                        ("Right Zone", (2 * orig_width // 3, 0, orig_width, orig_height))
                    ]
                elif "Horizontal" in zone_mode:
                    if num_floors == 1:
                        regions = [
                            ("Ground Floor", (0, 0, orig_width, orig_height))
                        ]
                    elif num_floors == 2:
                        regions = [
                            ("First Floor", (0, 0, orig_width, int(orig_height * floor_boundaries[0]))),
                            ("Ground Floor", (0, int(orig_height * floor_boundaries[0]), orig_width, orig_height))
                        ]
                    else:
                        regions = [
                            ("Second Floor", (0, 0, orig_width, int(orig_height * floor_boundaries[0]))),
                            ("First Floor", (0, int(orig_height * floor_boundaries[0]), orig_width, int(orig_height * floor_boundaries[1]))),
                            ("Ground Floor", (0, int(orig_height * floor_boundaries[1]), orig_width, orig_height))
                        ]
                else:
                    regions = [
                        ("Global Area", (0, 0, orig_width, orig_height))
                    ]

                # Find peaks and draw targeting reticles on final heatmap
                font_scale = max(0.5, (orig_width / 1920.0) * 0.7)
                thickness = max(1, int(orig_width / 1920.0 * 2))
                reticle_radius = max(25, int(orig_width / 1920.0 * 45))
                offset_y = max(30, int(orig_height * 0.04))

                for name, (rx1, ry1, rx2, ry2) in regions:
                    cropped = heatmap_blur[ry1:ry2, rx1:rx2]
                    if np.max(cropped) > 0:
                        _, _, _, max_loc = cv2.minMaxLoc(cropped)
                        hx = rx1 + max_loc[0]
                        hy = ry1 + max_loc[1]
                        
                        zone_hotspots.append({
                            "zone_name": name,
                            "x": hx,
                            "y": hy
                        })

                        # Draw red targeting reticle
                        cv2.circle(blended_heatmap, (hx, hy), reticle_radius, (0, 0, 255), thickness, cv2.LINE_AA)
                        cv2.circle(blended_heatmap, (hx, hy), max(3, int(reticle_radius * 0.1)), (0, 0, 255), -1)
                        
                        # Draw label
                        label_text = f"PEAK: {name.upper()}"
                        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        
                        cv2.rectangle(blended_heatmap, 
                                      (hx - int(w/2) - 8, hy - offset_y - h - 6), 
                                      (hx + int(w/2) + 8, hy - offset_y + 6), 
                                      (0, 0, 255), -1)
                        cv2.putText(blended_heatmap, label_text, 
                                    (hx - int(w/2), hy - offset_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # Draw Zone Boundaries and Labels on Final Heatmap
                if "Vertical" in zone_mode:
                    draw_dashed_line(blended_heatmap, (orig_width // 3, 0), (orig_width // 3, orig_height), (0, 165, 255), thickness)
                    draw_dashed_line(blended_heatmap, (2 * orig_width // 3, 0), (2 * orig_width // 3, orig_height), (0, 165, 255), thickness)
                    draw_zone_label(blended_heatmap, f"LEFT ZONE: {zone_occupancy['Left Zone']}", (orig_width // 6 - 50, 40), (0, 165, 255), font_scale, thickness)
                    draw_zone_label(blended_heatmap, f"CENTER ZONE: {zone_occupancy['Center Zone']}", (orig_width // 2 - 60, 40), (0, 165, 255), font_scale, thickness)
                    draw_zone_label(blended_heatmap, f"RIGHT ZONE: {zone_occupancy['Right Zone']}", (5 * orig_width // 6 - 60, 40), (0, 165, 255), font_scale, thickness)
                elif "Horizontal" in zone_mode:
                    # Draw floor labels directly on the final heatmap without grid divider lines
                    if num_floors == 1:
                        draw_zone_label(blended_heatmap, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(orig_height * 0.50)), (255, 165, 0), font_scale, thickness)
                    elif num_floors == 2:
                        draw_zone_label(blended_heatmap, f"FIRST FLOOR: {zone_occupancy['First Floor']}", (20, int(orig_height * (floor_boundaries[0] / 2))), (255, 165, 0), font_scale, thickness)
                        draw_zone_label(blended_heatmap, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(orig_height * ((floor_boundaries[0] + 1.0) / 2))), (255, 165, 0), font_scale, thickness)
                    else:
                        draw_zone_label(blended_heatmap, f"SECOND FLOOR: {zone_occupancy['Second Floor']}", (20, int(orig_height * (floor_boundaries[0] / 2))), (255, 165, 0), font_scale, thickness)
                        draw_zone_label(blended_heatmap, f"FIRST FLOOR: {zone_occupancy['First Floor']}", (20, int(orig_height * ((floor_boundaries[0] + floor_boundaries[1]) / 2))), (255, 165, 0), font_scale, thickness)
                        draw_zone_label(blended_heatmap, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(orig_height * ((floor_boundaries[1] + 1.0) / 2))), (255, 165, 0), font_scale, thickness)

                # ---------------------------------------------
                # VISUAL ATTENTION & ORIENTATION ANALYSIS
                # ---------------------------------------------
                # Resize the attention grid back to original size
                attention_high = cv2.resize(attention_accum, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
                
                # Normalize matrix
                max_att = np.max(attention_high)
                if max_att > 0:
                    attention_norm = np.clip(attention_high * (255.0 / max_att), 0, 255).astype(np.uint8)
                else:
                    attention_norm = attention_high.astype(np.uint8)
                
                # Smooth Gaussian blur (101x101) for attention glow
                attention_blur = cv2.GaussianBlur(attention_norm, (101, 101), 0)
                attention_color = cv2.applyColorMap(attention_blur, cv2.COLORMAP_JET)
                
                # Mask out low attention zones (less than 4% of peak)
                att_mask = attention_blur > (np.max(attention_blur) * 0.04)
                
                blended_attention = dimmed_frame.copy()
                if np.max(attention_blur) > 0:
                    # Blend 70% background, 30% attention heatmap
                    blended_attention[att_mask] = cv2.addWeighted(dimmed_frame, 0.70, attention_color, 0.30, 0)[att_mask]
                
                # Find the Top 3 Attention Peaks (Hotspots) with local non-maximum suppression
                attention_temp = attention_blur.copy()
                attention_peaks = []
                min_dist_px = int(orig_width * 0.12)  # Peak separation distance (~150px)
                
                total_accum_sum = np.sum(attention_accum)
                if total_accum_sum <= 0:
                    total_accum_sum = 1.0
                    
                peak_labels = ["Most Viewed Area", "Second Most Viewed Area", "Third Most Viewed Area"]
                # Colors for markers: BGR (Orange-Red, Gold, Purple/Cyan)
                marker_colors = [(0, 165, 255), (0, 215, 255), (255, 191, 0)]
                
                for rank in range(3):
                    if np.max(attention_temp) <= 0:
                        break
                    _, max_val, _, max_loc = cv2.minMaxLoc(attention_temp)
                    hx, hy = max_loc
                    
                    # Compute percentage of attention in a local 60px neighborhood
                    ghx, ghy = int(hx / grid_scale), int(hy / grid_scale)
                    radius_grid = int(60 / grid_scale)
                    
                    ghx_start = max(0, ghx - radius_grid)
                    ghx_end = min(grid_w, ghx + radius_grid + 1)
                    ghy_start = max(0, ghy - radius_grid)
                    ghy_end = min(grid_h, ghy + radius_grid + 1)
                    
                    peak_sum = 0
                    for gy in range(ghy_start, ghy_end):
                        for gx in range(ghx_start, ghx_end):
                            if np.sqrt((gx - ghx)**2 + (gy - ghy)**2) <= radius_grid:
                                peak_sum += attention_accum[gy, gx]
                                
                    percentage = (peak_sum / total_accum_sum) * 100.0
                    if percentage > 100.0:
                        percentage = 100.0
                        
                    attention_peaks.append({
                        "rank": rank + 1,
                        "label": peak_labels[rank],
                        "x": hx,
                        "y": hy,
                        "percentage": percentage,
                        "color": (marker_colors[rank][2], marker_colors[rank][1], marker_colors[rank][0])
                    })
                    
                    # Draw elegant map pointer pin
                    color = marker_colors[rank]
                    # Base target reticle
                    cv2.circle(blended_attention, (hx, hy), 8, color, thickness, cv2.LINE_AA)
                    cv2.circle(blended_attention, (hx, hy), 2, color, -1)
                    # Vertical pointer pin stem
                    cv2.line(blended_attention, (hx, hy - 40), (hx, hy), color, 2, cv2.LINE_AA)
                    # Outer round head of pin
                    cv2.circle(blended_attention, (hx, hy - 40), 12, color, -1)
                    # Inner white dot of pin head
                    cv2.circle(blended_attention, (hx, hy - 40), 4, (255, 255, 255), -1)
                    
                    # Elegant text badge at the top
                    label_text = f"{peak_labels[rank].upper()}: {percentage:.1f}%"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # Badge background rectangle
                    cv2.rectangle(blended_attention, 
                                  (hx - int(tw/2) - 8, hy - 58 - th - 6), 
                                  (hx + int(tw/2) + 8, hy - 58 + 6), 
                                  (12, 16, 23), -1)
                    # Badge border
                    cv2.rectangle(blended_attention, 
                                  (hx - int(tw/2) - 8, hy - 58 - th - 6), 
                                  (hx + int(tw/2) + 8, hy - 58 + 6), 
                                  color, 1, cv2.LINE_AA)
                    # Badge text
                    cv2.putText(blended_attention, label_text, 
                                (hx - int(tw/2), hy - 58), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    
                    # Suppress nearby detections
                    cv2.circle(attention_temp, (hx, hy), min_dist_px, 0, -1)

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
                    "zone_occupancy": zone_occupancy.copy(),
                    "zone_hotspots": zone_hotspots,
                    "blended_flow_map": blended_flow_map,
                    "pct_dominant": (dominant_count / 36.0) * 100.0,
                    "pct_dead": (dead_count / 36.0) * 100.0,
                    "pct_underutilized": (underutilized_count / 36.0) * 100.0,
                    "blended_attention": blended_attention,
                    "attention_peaks": attention_peaks
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
                    
                    # Render Zone occupancy metric and chart if enabled under the heatmap image
                    if "Disabled" not in zone_mode and r.get("zone_occupancy"):
                        st.markdown("---")
                        st.markdown("#### Zone Traffic Occupancy (Person-Frames)")
                        
                        peak_zone = max(r["zone_occupancy"], key=r["zone_occupancy"].get)
                        peak_count = r["zone_occupancy"][peak_zone]
                        
                        if peak_count > 0:
                            st.info(f"**Busiest Zone: {peak_zone.upper()}** ({peak_count} detections)")
                        else:
                            st.info("No zone occupancy recorded.")
                            
                        col_chart, _ = st.columns([6, 4])
                        with col_chart:
                            st.bar_chart(r["zone_occupancy"], height=180)

                with col_details:
                    st.markdown("#### Spatial Metrics Summary")
                    st.markdown(f"**Total Pedestrians Verified**: `{r['unique_visitors_count']}` unique tracks audited.")
                    
                    if r.get("zone_hotspots"):
                        st.info("Located Congestion Hotspots")
                        for zh in r["zone_hotspots"]:
                            st.markdown(f"* **{zh['zone_name']} Peak**: Coordinate: **X: {zh['x']}, Y: {zh['y']}**")

                st.markdown("---")
                
                # 2. Visitor Attention Hotspots Section
                st.markdown("### 2. Visitor Attention Hotspots")
                col_att_map, col_att_details = st.columns([6, 4])
                with col_att_map:
                    att_rgb = cv2.cvtColor(r["blended_attention"], cv2.COLOR_BGR2RGB)
                    st.image(att_rgb, caption=f"Visual Attention pointer hotspots map for {r['video_name']}", width="stretch")
                    
                with col_att_details:
                    st.markdown("#### Visual Attention Ranking")
                    st.markdown("Calculates shopper orientation and movement trails to estimate their visual attention peaks.")
                    
                    if r.get("attention_peaks"):
                        for peak in r["attention_peaks"]:
                            # Beautiful styled metric block
                            st.markdown(f"""
                            <div style="background-color: #151922; border-left: 5px solid rgb{peak['color']}; padding: 12px; border-radius: 4px; margin-bottom: 15px; border-top: 1px solid #222b3c; border-right: 1px solid #222b3c; border-bottom: 1px solid #222b3c;">
                                <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Rank {peak['rank']}</div>
                                <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff; margin-top: 4px;">{peak['label']}</div>
                                <div style="font-size: 1.8rem; font-weight: 800; color: rgb{peak['color']}; margin-top: 8px;">{peak['percentage']:.1f}% <span style="font-size: 0.9rem; font-weight: 400; color: #94a3b8;">of total visual attention</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No visual attention peaks detected in the video feed.")

                st.markdown("---")
                
                # 3. Floor Space Classification Section
                st.markdown("### 3. Floor Space Classification")
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
