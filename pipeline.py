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

        # Track unique visitors and their paths
        unique_visitors = set()
        path_history = defaultdict(list)  # track_id -> list of (x, y) coordinates
        sector_detections = defaultdict(int)

        # For blending the final heatmap
        reference_frame = first_frame.copy()

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

                    # Accumulate dwell intensity on the heatmap (heavier weight for staying still)
                    # Draw a soft radial peak (Gaussian-like) on the heatmap
                    cv2.circle(heatmap_accum, (foot_x, foot_y), 25, 1.5, -1)

                    # If tracked, perform tracker-specific operations
                    if track_ids is not None and idx < len(track_ids):
                        track_id = int(track_ids[idx])
                        current_active_ids.add(track_id)
                        unique_visitors.add(track_id)

                        # Store history path trail
                        path_history[track_id].append((foot_x, foot_y))
                        if len(path_history[track_id]) > 30:  # Keep last 30 coordinates for the trail
                            path_history[track_id].pop(0)

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
        try:
            cv2.destroyAllWindows()
        except cv2.error:  # headless OpenCV build
            pass

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


        print("SPATIAL ZONE CLASSIFICATION ANALYTICS:")
        print("-" * 40)
        print(f"Dominant Paths (High Traffic)   : {(dominant_count / 36.0) * 100.0:.1f}%")
        print(f"Underutilized Zones (Low Traffic): {(underutilized_count / 36.0) * 100.0:.1f}%")
        print(f"Dead Areas (Zero Traffic)       : {(dead_count / 36.0) * 100.0:.1f}%")
        print("-" * 40)

        # Save final static visual outputs
        flow_filename = f"flow_{base_name}.png"
        cv2.imwrite(heatmap_filename, blended_heatmap)
        cv2.imwrite(flow_filename, blended_flow_map)
        print(f"[SUCCESS] Saved movement heatmap image to '{heatmap_filename}'")
        print(f"[SUCCESS] Saved spatial zone classification image to '{flow_filename}'")
        print(f"[SUCCESS] Saved full annotated live tracking video to '{output_filename}'")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
