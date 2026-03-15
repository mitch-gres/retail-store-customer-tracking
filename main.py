import cv2
from ultralytics import YOLO
import supervision as sv
import time
import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import os 

# ==========================================
# 🎛️ MAIN CONTROL PANEL
# ==========================================
# 1. INPUT SOURCE
USE_LIVE_CAMERA = False          
CAMERA_INDEX = 1                 
VIDEO_PATH = "walking_hallway_3.mp4" 

# 2. OUTPUT & DISPLAY
SHOW_LIVE_VIDEO = True           
SAVE_OUTPUT_VIDEO = True         
PROCESS_EVERY_N_FRAMES = 5      

# 3. AI & TRACKING THRESHOLDS
SIMILARITY_THRESHOLD = 0.65      
MIN_BOX_AREA = 8000              
MAX_OVERLAP = 0.05               

# 4. FEATURE GALLERY SETTINGS
MAX_GALLERY_SIZE = 5             
GALLERY_UPDATE_THRESHOLD = 0.8  

# 5. TIME TRACKING DISPLAY (NEW)
TIME_WARNING_THRESHOLD = 30      # Seconds in aisle before text turns YELLOW
TIME_ALERT_THRESHOLD = 60        # Seconds in aisle before text turns RED
AISLE_RESET_TIMEOUT = 3.0        # Seconds out of frame before "Aisle Time" resets
# ==========================================

# --- HELPER FUNCTION: Get Clothing Fingerprint ---
def get_embedding(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10: 
        return None
    
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(crop_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat = feature_extractor(tensor).squeeze() 
    
    return F.normalize(feat, p=2, dim=0)

# --- HELPER FUNCTION: Calculate Box Overlap (IoU) ---
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0: return 0
    return inter_area / union_area

# --- 1. SETUP VIDEO INPUT ---
if USE_LIVE_CAMERA:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print("📷 Using Live Camera Feed...")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    print(f"🎬 Using Video File: {VIDEO_PATH}")

if not cap.isOpened():
    print("❌ ERROR: Could not open video source.")
    exit()

# --- 2. SETUP VIDEO WRITER ---
if SAVE_OUTPUT_VIDEO:
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if original_fps == 0: original_fps = 30 
    
    output_fps = max(1, original_fps // PROCESS_EVERY_N_FRAMES)
    
    if USE_LIVE_CAMERA:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"live_tracking_{timestamp}.mp4"
    else:
        base_name, ext = os.path.splitext(VIDEO_PATH)
        output_path = f"{base_name}_tracked{ext}"
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (original_width, original_height))

# --- 3. SETUP AI MODELS ---
model = YOLO("yolov8x-seg.pt") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
feature_extractor.eval() 

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)), 
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. STORAGE & DATABASES ---
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# --- TIME TRACKING DICTS ---
store_entry_times = {}   
aisle_entry_times = {}   
last_seen_times = {}     
aisle_visit_counts = {}  

customer_database = {} 
id_map = {} 
next_customer_id = 1
reid_match_scores = {} 

frame_count = 0
start_time = time.time()

print("🚀 Starting processing... Please wait.")

# --- 5. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame_count += 1
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue

    frame_height, frame_width = frame.shape[:2]

    results = model.track(frame, classes=[0], persist=True, tracker="botsort.yaml", verbose=False)[0] 
    detections = sv.Detections.from_ultralytics(results)

    labels = []
    currently_in_frame_pids = set() 

    if detections.tracker_id is not None:
        current_time = time.time() 
        
        mask = detections.area >= MIN_BOX_AREA
        detections = detections[mask]
        
        occluded_indices = set()
        if len(detections.xyxy) > 1:
            for i in range(len(detections.xyxy)):
                for j in range(i + 1, len(detections.xyxy)):
                    if calculate_iou(detections.xyxy[i], detections.xyxy[j]) > MAX_OVERLAP:
                        occluded_indices.add(i)
                        occluded_indices.add(j)

        for i, yolo_id in enumerate(detections.tracker_id):
            bbox = detections.xyxy[i]
            yolo_conf = detections.confidence[i] if detections.confidence is not None else 0.0
            is_occluded = i in occluded_indices
            
            hud_text = "Processing..."
            
            # 1. NEW YOLO ID DETECTED
            if yolo_id not in id_map:
                if is_occluded:
                    hud_text = "WAITING... (Occluded)"
                else:
                    embedding = get_embedding(frame, bbox)
                    if embedding is not None:
                        best_match_id = None
                        best_similarity = -1
                        
                        for known_id, gallery in customer_database.items():
                            for known_embedding in gallery:
                                similarity = F.cosine_similarity(embedding.unsqueeze(0), known_embedding.unsqueeze(0)).item()
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match_id = known_id

                        if best_similarity > SIMILARITY_THRESHOLD:
                            id_map[yolo_id] = best_match_id
                            reid_match_scores[best_match_id] = best_similarity
                        else:
                            persistent_id = f"CUST-{next_customer_id}"
                            id_map[yolo_id] = persistent_id
                            customer_database[persistent_id] = [embedding] 
                            
                            store_entry_times[persistent_id] = current_time
                            aisle_entry_times[persistent_id] = current_time
                            last_seen_times[persistent_id] = current_time
                            aisle_visit_counts[persistent_id] = 1  
                            
                            reid_match_scores[persistent_id] = 1.0 
                            next_customer_id += 1
                    else:
                        hud_text = "Crop Error"

            # 2. EXISTING CUSTOMER
            elif yolo_id in id_map:
                p_id = id_map[yolo_id]
                currently_in_frame_pids.add(p_id) 
                
                if current_time - last_seen_times.get(p_id, current_time) > AISLE_RESET_TIMEOUT:
                    aisle_entry_times[p_id] = current_time
                    aisle_visit_counts[p_id] = aisle_visit_counts.get(p_id, 0) + 1 
                last_seen_times[p_id] = current_time
                
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                margin_x = int(frame_width * 0.10) 
                margin_y = int(frame_height * 0.10) 
                
                is_center_frame = (margin_x < center_x < (frame_width - margin_x) and 
                                   margin_y < center_y < (frame_height - margin_y))
                
                margin = 50
                is_fully_in_frame = (bbox[0] > margin and bbox[1] > margin and bbox[2] < (frame_width - margin) and bbox[3] < (frame_height - margin))
                
                if is_fully_in_frame and not is_occluded and is_center_frame:
                    current_embedding = get_embedding(frame, bbox)
                    if current_embedding is not None:
                        gallery = customer_database[p_id]
                        max_sim_to_gallery = max([F.cosine_similarity(current_embedding.unsqueeze(0), k_emb.unsqueeze(0)).item() for k_emb in gallery])
                        
                        if max_sim_to_gallery < GALLERY_UPDATE_THRESHOLD:
                            gallery.append(current_embedding)
                            if len(gallery) > MAX_GALLERY_SIZE:
                                gallery.pop(0) 
                                
                        reid_match_scores[p_id] = max_sim_to_gallery

                reid_score = reid_match_scores.get(p_id, 1.0)
                occ_marker = "*" if is_occluded else ""
                
                gallery_size = len(customer_database.get(p_id, []))
                edge_marker = "" if is_center_frame else " (EDGE)"
                
                hud_text = f"{p_id}{occ_marker}{edge_marker} | Angles: {gallery_size} | YOLO:{yolo_conf:.2f} | ReID:{reid_score:.2f}"
                
            labels.append(hud_text)

    annotated_frame = frame.copy()
    if len(detections) > 0:
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        if labels:
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # ==========================================
    # ⏱️ DRAW TOP-RIGHT TIME TRACKING HUD
    # ==========================================
    current_time = time.time()
    y_offset = 40 
    
    for p_id in sorted(list(currently_in_frame_pids)):
        store_time = current_time - store_entry_times.get(p_id, current_time)
        aisle_time = current_time - aisle_entry_times.get(p_id, current_time)
        visit_count = aisle_visit_counts.get(p_id, 1) 
        
        # --- NEW COLOR LOGIC ---
        if aisle_time > TIME_ALERT_THRESHOLD:
            text_color = (0, 0, 255) # Red (Highest priority)
        elif aisle_time > TIME_WARNING_THRESHOLD or visit_count > 1:
            text_color = (0, 255, 255) # Yellow (If warning time hit OR multiple visits)
        else:
            text_color = (255, 255, 255) # White
        # -----------------------
            
        display_text = f"{p_id} | Visits: {visit_count} | Aisle: {aisle_time:.1f}s | Store: {store_time:.1f}s"
        
        (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x_pos = frame_width - text_width - 20
        
        cv2.rectangle(annotated_frame, (x_pos - 10, y_offset - text_height - 10), (x_pos + text_width + 10, y_offset + 10), (0, 0, 0), -1)
        cv2.putText(annotated_frame, display_text, (x_pos, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        y_offset += 45 
    # ==========================================

    if SAVE_OUTPUT_VIDEO:
        out.write(annotated_frame)

    if SHOW_LIVE_VIDEO:
        scale_factor = 0.5 
        display_width = int(frame_width * scale_factor)
        display_height = int(frame_height * scale_factor)
        display_frame = cv2.resize(annotated_frame, (display_width, display_height))

        cv2.imshow("Store Tracking POC", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

# --- AUTOMATIC SUMMARY REPORT ---
total_time = time.time() - start_time
print("\n" + "="*40)
print("📊 FEATURE GALLERY SUMMARY")
print("="*40)
print(f"⏱️  Processing Time:  {total_time:.1f} seconds")
print(f"🎞️  Frames Processed: {frame_count // PROCESS_EVERY_N_FRAMES}")
print(f"👥 Total Profiles Created: {len(customer_database)}")

cap.release()
if SAVE_OUTPUT_VIDEO:
    out.release()
cv2.destroyAllWindows()