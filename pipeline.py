import cv2
import numpy as np
import argparse
import sys
import time
import os
from collections import defaultdict
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="CCTV Business Intelligence and Physical-World Memory POC")
    parser.add_argument("--input", type=str, default="0", help="Comma-separated paths to input video files or webcam indices (e.g. '0,lobby.mp4')")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model version (default: yolov8n.pt)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracking algorithm config (default: bytetrack.yaml)")
    parser.add_argument("--zones", type=str, default="vertical", choices=["vertical", "horizontal", "disabled"], help="Zone division mode: vertical (Left/Center/Right), horizontal (Floors), or disabled (default: vertical)")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu", "npu"], help="Target acceleration device: cpu, gpu, npu (default: gpu)")
    parser.add_argument("--skip", type=int, default=1, help="Frame skipping interval (default: 1, i.e. process every frame)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Map device parameter
    device_opt = args.device.upper() # "CPU", "GPU", "NPU"
    
    # Load YOLO Model
    print(f"[INFO] Initializing hardware-accelerated neural networks...")
    try:
        import torch
        
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

        if device_opt == "GPU":
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
                device_str = "cpu"
                detected_device_name = "CPU (No GPU acceleration detected)"
        elif device_opt == "NPU":
            if has_ov_npu:
                device_str = "intel:npu"
                use_openvino = True
                detected_device_name = "Intel AI Boost NPU"
            else:
                print("[WARNING] Intel AI Boost NPU was requested but no NPU device was detected. Falling back to CPU...")
                device_str = "cpu"
                detected_device_name = "CPU (Fallback)"
        else:
            device_str = "cpu"
            detected_device_name = "CPU (Standard)"

        print(f"[INFO] Targeting execution device: {detected_device_name}")

        if use_openvino:
            # Derive OpenVINO model directory name from model file name (e.g. yolov8n_openvino_model)
            model_base = os.path.splitext(args.model)[0]
            model_dir = f"{model_base}_openvino_model"
            
            if not os.path.exists(model_dir):
                print(f"[INFO] First-time setup: Exporting {args.model} to OpenVINO format for hardware acceleration...")
                base_model = YOLO(args.model)
                base_model.export(format="openvino")
                
            model = YOLO(model_dir)
            track_device = device_str
        else:
            model = YOLO(args.model)
            track_device = device_str
    except Exception as e:
        print(f"[WARNING] Failed to load YOLO model with hardware acceleration: {e}")
        print("[INFO] Falling back to standard CPU model...")
        try:
            model = YOLO(args.model)
            track_device = "cpu"
        except Exception as fallback_err:
            print(f"[ERROR] Failed to load fallback model: {fallback_err}")
            sys.exit(1)
    # Parse inputs (comma-separated list)
    sources = [s.strip().strip('"').strip("'") for s in args.input.split(",")]

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
        cv2.rectangle(img, (pt[0] - 5, pt[1] - h - 5), (pt[0] + w + 5, pt[1] + 5), (15, 15, 15), -1)
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

    for source_idx, source_str in enumerate(sources):
        source = int(source_str) if source_str.isdigit() else source_str
        
        if isinstance(source, int):
            base_name = f"cam{source}"
            print(f"\n[INFO] Processing Source {source_idx + 1}/{len(sources)}: Accessing webcam {source}...")
        else:
            base_name = os.path.splitext(os.path.basename(source))[0]
            print(f"\n[INFO] Processing Source {source_idx + 1}/{len(sources)}: Opening video file: {source_str}...")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {source_str}")
            continue

        # Read first frame to get dimensions and initialize heatmap accumulator
        ret, first_frame = cap.read()
        if not ret:
            print(f"[ERROR] Failed to read from video source: {source_str}")
            cap.release()
            continue

        height, width, _ = first_frame.shape
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        print(f"[INFO] Resolution: {width}x{height} | FPS: {fps}")

        # Heatmap Accumulator (float32 matrix to store dwell intensity)
        heatmap_accum = np.zeros((height, width), dtype=np.float32)
        attention_accum = np.zeros((height, width), dtype=np.float32)

        # Track unique visitors and their paths
        unique_visitors = set()
        path_history = defaultdict(list)  # track_id -> list of (x, y) coordinates
        sector_detections = defaultdict(int)

        # For blending the final heatmap
        reference_frame = first_frame.copy()

        # Initialize floor splits to defaults (will be updated dynamically at completion)
        num_floors = 3
        floor_boundaries = [0.20, 0.50]
        all_detected_y_coords = []
        floors = ["Second Floor", "First Floor", "Ground Floor"]

        # Initialize Zone Occupancy Counts (Cumulative presence)
        zone_occupancy = {}
        if args.zones == "vertical":
            zone_occupancy = {"Left Zone": 0, "Center Zone": 0, "Right Zone": 0}
        elif args.zones == "horizontal":
            zone_occupancy = {f: 0 for f in floors}

        # Set up video writer
        output_filename = f"output_{base_name}.mp4"
        heatmap_filename = f"heatmap_{base_name}.png"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        print(f"[INFO] Starting processing for {base_name}. Press 'q' in the window to finish early.")

        # Reset capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Color palette for bounding boxes based on Track ID
        def get_color(track_id):
            np.random.seed(int(track_id))
            return [int(c) for c in np.random.randint(0, 255, size=3)]

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            frame_count += 1
            
            # Perfect skip interval
            if frame_count % args.skip != 0:
                ret = cap.grab()
                if not ret:
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break

            current_active_ids = set()

            # Run tracking using YOLO + ByteTrack
            # class 0 represents 'person' in the COCO dataset
            tracker_config = args.tracker
            if tracker_config == "bytetrack.yaml":
                custom_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_tracker.yaml")
                if os.path.exists(custom_config):
                    tracker_config = custom_config

            results = model.track(
                source=frame,
                persist=True,
                classes=[0],
                conf=args.confidence,
                tracker=tracker_config,
                verbose=False,
                device=track_device
            )

            # Process tracking results
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else None
                confidences = results[0].boxes.conf.cpu().numpy()

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    conf = confidences[idx]

                    # Determine the track center-bottom (foot position) for paths and heatmaps
                    foot_x = int((x1 + x2) / 2)
                    foot_y = y2

                    # Log visitor count into spatial sector grid (cumulative presence)
                    sr = min(5, int(foot_y / (height / 6.0)))
                    sc = min(5, int(foot_x / (width / 6.0)))
                    sector_detections[(sr, sc)] += 1
                    all_detected_y_coords.append(foot_y)

                    # Accumulate dwell intensity on the heatmap (heavier weight for staying still)
                    # Draw a soft radial peak (Gaussian-like) on the heatmap
                    cv2.circle(heatmap_accum, (foot_x, foot_y), 25, 1.5, -1)

                    # Increment zone/floor occupancy counters
                    if args.zones == "vertical":
                        if foot_x < width // 3:
                            zone_occupancy["Left Zone"] += 1
                        elif foot_x < 2 * width // 3:
                            zone_occupancy["Center Zone"] += 1
                        else:
                            zone_occupancy["Right Zone"] += 1
                    elif args.zones == "horizontal":
                        if args.num_floors == 1:
                            zone_occupancy["Ground Floor"] += 1
                        elif args.num_floors == 2:
                            if foot_y < int(height * floor_boundaries[0]):
                                zone_occupancy["First Floor"] += 1
                            else:
                                zone_occupancy["Ground Floor"] += 1
                        else:
                            if foot_y < int(height * floor_boundaries[0]):
                                zone_occupancy["Second Floor"] += 1
                            elif foot_y < int(height * floor_boundaries[1]):
                                zone_occupancy["First Floor"] += 1
                            else:
                                zone_occupancy["Ground Floor"] += 1

                    # If tracked, perform tracker-specific operations
                    if track_ids is not None and idx < len(track_ids):
                        track_id = int(track_ids[idx])
                        current_active_ids.add(track_id)
                        unique_visitors.add(track_id)

                        # Store history path trail
                        path_history[track_id].append((foot_x, foot_y))
                        if len(path_history[track_id]) > 30:  # Keep last 30 coordinates for the trail
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
                                sample_x = int(hx + vx * (step * 20))
                                sample_y = int(hy + vy * (step * 20))
                                if 0 <= sample_y < height and 0 <= sample_x < width:
                                    # Decay weight slightly as distance from observer increases
                                    weight = 1.5 - (step * 0.1)
                                    attention_accum[sample_y, sample_x] += weight
                        else:
                            # Person is standing/dwelling: project a circular scan field around their position
                            for dy_px in range(-30, 31, 10):
                                for dx_px in range(-30, 31, 10):
                                    dist = np.sqrt(dx_px**2 + dy_px**2)
                                    if dist <= 30:
                                        sample_x = int(hx + dx_px)
                                        sample_y = int(hy + dy_px)
                                        if 0 <= sample_y < height and 0 <= sample_x < width:
                                            attention_accum[sample_y, sample_x] += 1.0

                        # Draw track trail (spaghetti lines)
                        trail = path_history[track_id]
                        for i in range(1, len(trail)):
                            cv2.line(frame, trail[i-1], trail[i], get_color(track_id), 2, cv2.LINE_AA)

                        # Draw elegant bounding box
                        color = get_color(track_id)
                        # Bounding box corners style
                        box_thickness = 2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                        
                        # Tag label
                        label = f"Shopper #{track_id}"
                        # Background tag box
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        # Draw untracked box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

            # Draw Zone Separator Lines and Labels on Live Frame
            font_scale = max(0.5, (width / 1920.0) * 0.7)
            thickness = max(1, int(width / 1920.0 * 2))
            
            if args.zones == "vertical":
                draw_dashed_line(frame, (width // 3, 0), (width // 3, height), (0, 165, 255), thickness)
                draw_dashed_line(frame, (2 * width // 3, 0), (2 * width // 3, height), (0, 165, 255), thickness)
                draw_zone_label(frame, f"LEFT ZONE: {zone_occupancy['Left Zone']}", (width // 6 - 50, 40), (0, 165, 255), font_scale, thickness)
                draw_zone_label(frame, f"CENTER ZONE: {zone_occupancy['Center Zone']}", (width // 2 - 60, 40), (0, 165, 255), font_scale, thickness)
                draw_zone_label(frame, f"RIGHT ZONE: {zone_occupancy['Right Zone']}", (5 * width // 6 - 60, 40), (0, 165, 255), font_scale, thickness)
            elif args.zones == "horizontal":
                # Draw floor labels directly on the live frame without grid divider lines
                if args.num_floors == 1:
                    draw_zone_label(frame, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(height * 0.50)), (255, 165, 0), font_scale, thickness)
                elif args.num_floors == 2:
                    draw_zone_label(frame, f"FIRST FLOOR: {zone_occupancy['First Floor']}", (20, int(height * (floor_boundaries[0] / 2))), (255, 165, 0), font_scale, thickness)
                    draw_zone_label(frame, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(height * ((floor_boundaries[0] + 1.0) / 2))), (255, 165, 0), font_scale, thickness)
                else:
                    draw_zone_label(frame, f"SECOND FLOOR: {zone_occupancy['Second Floor']}", (20, int(height * (floor_boundaries[0] / 2))), (255, 165, 0), font_scale, thickness)
                    draw_zone_label(frame, f"FIRST FLOOR: {zone_occupancy['First Floor']}", (20, int(height * ((floor_boundaries[0] + floor_boundaries[1]) / 2))), (255, 165, 0), font_scale, thickness)
                    draw_zone_label(frame, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(height * ((floor_boundaries[1] + 1.0) / 2))), (255, 165, 0), font_scale, thickness)

            # Draw a futuristic UI HUD (Heads-Up Display) overlay directly on the video
            # Semi-transparent overlay block for status
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 110), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Live Status Indicators
            cv2.circle(frame, (25, 30), 6, (0, 255, 0), -1) # Green pulsing dot
            cv2.putText(frame, f"LIVE CCTV FEED - {base_name.upper()}", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # Analytics counts
            cv2.putText(frame, f"TOTAL CUSTOMERS: {len(unique_visitors)}", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"ACTIVE SHOPPERS: {len(current_active_ids)}", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # Render current frame rate
            curr_fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {curr_fps:.1f}", (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Write frame to output video file
            writer.write(frame)

            # Show live video window (catch errors in headless environments like Google Colab)
            try:
                cv2.imshow("CCTV Intelligence POC (Press 'q' to Exit Current Source)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                pass

        # Clean up stream
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Automatically detect atrium floor levels from accumulated shopper vertical density
        if args.zones == "horizontal":
            floor_boundaries, floors = detect_atrium_floors(all_detected_y_coords, height)
            args.num_floors = len(floors)
            
            # Recalculate zone_occupancy with the exact auto-detected splits
            zone_occupancy = {f: 0 for f in floors}
            for y in all_detected_y_coords:
                if len(floor_boundaries) == 0:
                    zone_occupancy["Ground Floor"] += 1
                elif len(floor_boundaries) == 1:
                    if y < int(height * floor_boundaries[0]):
                        zone_occupancy["First Floor"] += 1
                    else:
                        zone_occupancy["Ground Floor"] += 1
                else:
                    if y < int(height * floor_boundaries[0]):
                        zone_occupancy["Second Floor"] += 1
                    elif y < int(height * floor_boundaries[1]):
                        zone_occupancy["First Floor"] += 1
                    else:
                        zone_occupancy["Ground Floor"] += 1

        print("\n" + "="*50)
        print(f"STREAM COMPLETED FOR {base_name.upper()} - COMPILING POC ANALYTICS")
        print("="*50)
        print(f"Total Unique Shoppers Tracked: {len(unique_visitors)}")

        # 1. GENERATE MOVEMENT HEATMAP
        # Normalize heatmap matrix to range 0-255
        if np.max(heatmap_accum) > 0:
            heatmap_norm = np.clip(heatmap_accum * (255.0 / np.max(heatmap_accum)), 0, 255).astype(np.uint8)
        else:
            heatmap_norm = heatmap_accum.astype(np.uint8)

        # Apply Gaussian blur for smooth density cloud
        heatmap_blur = cv2.GaussianBlur(heatmap_norm, (35, 35), 0)

        # Colorize using JET color map (Blue is low activity, Red is high activity)
        heatmap_color = cv2.applyColorMap(heatmap_blur, cv2.COLORMAP_JET)

        # Apply transparency mask so areas with zero movement show original video background
        mask = heatmap_blur > 5
        blended_heatmap = reference_frame.copy()
        blended_heatmap[mask] = cv2.addWeighted(reference_frame, 0.4, heatmap_color, 0.6, 0)[mask]

        # Determine zone regions for separate peak checking
        zone_hotspots = []
        
        if args.zones == "vertical":
            regions = [
                ("Left Zone", (0, 0, width // 3, height)),
                ("Center Zone", (width // 3, 0, 2 * width // 3, height)),
                ("Right Zone", (2 * width // 3, 0, width, height))
            ]
        elif args.zones == "horizontal":
            if args.num_floors == 1:
                regions = [
                    ("Ground Floor", (0, 0, width, height))
                ]
            elif args.num_floors == 2:
                regions = [
                    ("First Floor", (0, 0, width, int(height * floor_boundaries[0]))),
                    ("Ground Floor", (0, int(height * floor_boundaries[0]), width, height))
                ]
            else:
                regions = [
                    ("Second Floor", (0, 0, width, int(height * floor_boundaries[0]))),
                    ("First Floor", (0, int(height * floor_boundaries[0]), width, int(height * floor_boundaries[1]))),
                    ("Ground Floor", (0, int(height * floor_boundaries[1]), width, height))
                ]
        else:
            regions = [
                ("Global Area", (0, 0, width, height))
            ]

        # Find peaks and draw targeting reticles on final heatmap
        reticle_radius = max(25, int(width / 1920.0 * 45))
        offset_y = max(30, int(height * 0.04))

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
        if args.zones == "vertical":
            draw_dashed_line(blended_heatmap, (width // 3, 0), (width // 3, height), (0, 165, 255), thickness)
            draw_dashed_line(blended_heatmap, (2 * width // 3, 0), (2 * width // 3, height), (0, 165, 255), thickness)
            draw_zone_label(blended_heatmap, f"LEFT ZONE: {zone_occupancy['Left Zone']}", (width // 6 - 50, 40), (0, 165, 255), font_scale, thickness)
            draw_zone_label(blended_heatmap, f"CENTER ZONE: {zone_occupancy['Center Zone']}", (width // 2 - 60, 40), (0, 165, 255), font_scale, thickness)
            draw_zone_label(blended_heatmap, f"RIGHT ZONE: {zone_occupancy['Right Zone']}", (5 * width // 6 - 60, 40), (0, 165, 255), font_scale, thickness)
        elif args.zones == "horizontal":
            # Draw floor labels directly on the final heatmap without grid divider lines
            if args.num_floors == 1:
                draw_zone_label(blended_heatmap, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(height * 0.50)), (255, 165, 0), font_scale, thickness)
            elif args.num_floors == 2:
                draw_zone_label(blended_heatmap, f"FIRST FLOOR: {zone_occupancy['First Floor']}", (20, int(height * (floor_boundaries[0] / 2))), (255, 165, 0), font_scale, thickness)
                draw_zone_label(blended_heatmap, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(height * ((floor_boundaries[0] + 1.0) / 2))), (255, 165, 0), font_scale, thickness)
            else:
                draw_zone_label(blended_heatmap, f"SECOND FLOOR: {zone_occupancy['Second Floor']}", (20, int(height * (floor_boundaries[0] / 2))), (255, 165, 0), font_scale, thickness)
                draw_zone_label(blended_heatmap, f"FIRST FLOOR: {zone_occupancy['First Floor']}", (20, int(height * ((floor_boundaries[0] + floor_boundaries[1]) / 2))), (255, 165, 0), font_scale, thickness)
                draw_zone_label(blended_heatmap, f"GROUND FLOOR: {zone_occupancy['Ground Floor']}", (20, int(height * ((floor_boundaries[1] + 1.0) / 2))), (255, 165, 0), font_scale, thickness)

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
        
        font_scale = max(0.4, (width / 1920.0) * 0.55)
        thickness = max(1, int(width / 1920.0 * 2))
        
        for sr in range(6):
            for sc in range(6):
                count = sector_detections[(sr, sc)]
                ratio = count / max_sector_detections if max_sector_detections > 0 else 0
                
                # Sector pixel coordinates
                x1_sec = int(sc * (width / 6.0))
                y1_sec = int(sr * (height / 6.0))
                x2_sec = int((sc + 1) * (width / 6.0))
                y2_sec = int((sr + 1) * (height / 6.0))
                
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



        # Print final zone traffic metrics to the terminal
        if args.zones != "disabled" and zone_occupancy:
            print("\nZONE TRAFFIC OCCUPANCY (Person-Frames):")
            print("-" * 40)
            for zone_name, count in zone_occupancy.items():
                print(f"{zone_name:20s}: {count} detections")
            print("-" * 40)
            
            # Print separate floor/zone peak locations in terminal
            print("LOCATED CONGESTION HOTSPOTS:")
            for zh in zone_hotspots:
                print(f"* {zh['zone_name']:20s} Peak: X={zh['x']}, Y={zh['y']}")
            print("-" * 40)
            
            peak_zone = max(zone_occupancy, key=zone_occupancy.get)
            peak_count = zone_occupancy[peak_zone]
            if peak_count > 0:
                print(f"BUSIEST AREA: {peak_zone.upper()} with {peak_count} cumulative detections.")
            else:
                print("No zone activity recorded.")
            print("-" * 40)

        # ---------------------------------------------
        # VISUAL ATTENTION & ORIENTATION ANALYSIS
        # ---------------------------------------------
        # Normalize matrix
        max_att = np.max(attention_accum)
        if max_att > 0:
            attention_norm = np.clip(attention_accum * (255.0 / max_att), 0, 255).astype(np.uint8)
        else:
            attention_norm = attention_accum.astype(np.uint8)
        
        # Smooth Gaussian blur (101x101) for attention glow
        attention_blur = cv2.GaussianBlur(attention_norm, (101, 101), 0)
        attention_color = cv2.applyColorMap(attention_blur, cv2.COLORMAP_JET)
        
        # Mask out low attention zones (less than 4% of peak)
        att_mask = attention_blur > (np.max(attention_blur) * 0.04)
        
        dimmed_frame = cv2.convertScaleAbs(reference_frame, alpha=0.95, beta=0)
        blended_attention = dimmed_frame.copy()
        if np.max(attention_blur) > 0:
            blended_attention[att_mask] = cv2.addWeighted(dimmed_frame, 0.70, attention_color, 0.30, 0)[att_mask]
        
        # Find the Top 3 Attention Peaks (Hotspots) with local non-maximum suppression
        attention_temp = attention_blur.copy()
        attention_peaks = []
        min_dist_px = int(width * 0.12)  # Peak separation distance (~150px)
        
        total_accum_sum = np.sum(attention_accum)
        if total_accum_sum <= 0:
            total_accum_sum = 1.0
            
        peak_labels = ["Most Viewed Area", "Second Most Viewed Area", "Third Most Viewed Area"]
        marker_colors = [(0, 165, 255), (0, 215, 255), (255, 191, 0)] # BGR
        
        # Font scales for drawing
        font_scale = max(0.5, (width / 1920.0) * 0.7)
        thickness = max(1, int(width / 1920.0 * 2))
        
        for rank in range(3):
            if np.max(attention_temp) <= 0:
                break
            _, max_val, _, max_loc = cv2.minMaxLoc(attention_temp)
            hx, hy = max_loc
            
            # Compute percentage of attention in a local 60px neighborhood
            ghx_start = max(0, hx - 60)
            ghx_end = min(width, hx + 61)
            ghy_start = max(0, hy - 60)
            ghy_end = min(height, hy + 61)
            
            peak_sum = 0
            for gy in range(ghy_start, ghy_end):
                for gx in range(ghx_start, ghx_end):
                    if np.sqrt((gx - hx)**2 + (gy - hy)**2) <= 60:
                        peak_sum += attention_accum[gy, gx]
                        
            percentage = (peak_sum / total_accum_sum) * 100.0
            if percentage > 100.0:
                percentage = 100.0
                
            attention_peaks.append({
                "rank": rank + 1,
                "label": peak_labels[rank],
                "x": hx,
                "y": hy,
                "percentage": percentage
            })
            
            # Draw map pointer pin
            color = marker_colors[rank]
            cv2.circle(blended_attention, (hx, hy), 8, color, thickness, cv2.LINE_AA)
            cv2.circle(blended_attention, (hx, hy), 2, color, -1)
            cv2.line(blended_attention, (hx, hy - 40), (hx, hy), color, 2, cv2.LINE_AA)
            cv2.circle(blended_attention, (hx, hy - 40), 12, color, -1)
            cv2.circle(blended_attention, (hx, hy - 40), 4, (255, 255, 255), -1)
            
            label_text = f"{peak_labels[rank].upper()}: {percentage:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            cv2.rectangle(blended_attention, 
                          (hx - int(tw/2) - 8, hy - 58 - th - 6), 
                          (hx + int(tw/2) + 8, hy - 58 + 6), 
                          (12, 16, 23), -1)
            cv2.rectangle(blended_attention, 
                          (hx - int(tw/2) - 8, hy - 58 - th - 6), 
                          (hx + int(tw/2) + 8, hy - 58 + 6), 
                          color, 1, cv2.LINE_AA)
            cv2.putText(blended_attention, label_text, 
                        (hx - int(tw/2), hy - 58), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Suppress nearby detections
            cv2.circle(attention_temp, (hx, hy), min_dist_px, 0, -1)

        print("SPATIAL ZONE CLASSIFICATION ANALYTICS:")
        print("-" * 40)
        print(f"Dominant Paths (High Traffic)   : {(dominant_count / 36.0) * 100.0:.1f}%")
        print(f"Underutilized Zones (Low Traffic): {(underutilized_count / 36.0) * 100.0:.1f}%")
        print(f"Dead Areas (Zero Traffic)       : {(dead_count / 36.0) * 100.0:.1f}%")
        print("-" * 40)

        print("VISUAL ATTENTION RANKING ANALYTICS:")
        print("-" * 40)
        for peak in attention_peaks:
            print(f"Rank {peak['rank']}: {peak['label']:25s} | {peak['percentage']:.1f}% of attention | Peak: X={peak['x']}, Y={peak['y']}")
        print("-" * 40)

        # Save final static visual outputs
        flow_filename = f"flow_{base_name}.png"
        attention_filename = f"attention_{base_name}.png"
        cv2.imwrite(heatmap_filename, blended_heatmap)
        cv2.imwrite(flow_filename, blended_flow_map)
        cv2.imwrite(attention_filename, blended_attention)
        print(f"[SUCCESS] Saved movement heatmap image to '{heatmap_filename}'")
        print(f"[SUCCESS] Saved spatial zone classification image to '{flow_filename}'")
        print(f"[SUCCESS] Saved visual attention hotspots image to '{attention_filename}'")
        print(f"[SUCCESS] Saved full annotated live tracking video to '{output_filename}'")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
