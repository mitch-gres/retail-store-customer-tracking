import cv2
from ultralytics import YOLO
import supervision as sv
import time
import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

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
video_path = "walking_hallway.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ ERROR: Could not open video file at '{video_path}'.")
    exit()

# --- 2. SETUP AI MODELS ---
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

# --- 3. STORAGE & DATABASES ---
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

entry_times = {}
# customer_database is now a dictionary of LISTS: { "CUST-1": [emb1, emb2, emb3] }
customer_database = {} 
id_map = {} 
next_customer_id = 1
reid_match_scores = {} 

# --- 4. MAIN LOOP PARAMETERS ---
PROCESS_EVERY_N_FRAMES = 10
SHOW_LIVE_VIDEO = True

SIMILARITY_THRESHOLD = 0.55
MIN_BOX_AREA = 8000 
MAX_OVERLAP = 0.10  

# --- NEW GALLERY PARAMETERS ---
MAX_GALLERY_SIZE = 5         # Maximum number of angles to save per person
GALLERY_UPDATE_THRESHOLD = 0.85 # If a frame is 85% similar to a saved angle, don't save it. If it's lower, save as a new angle!

frame_count = 0
start_time = time.time()

print("🚀 Starting accelerated processing with Feature Gallery... Please wait.")

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
                        
                        # FEATURE GALLERY MATCHING LOGIC
                        # Check the new crop against EVERY saved angle for EVERY customer
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
                            # Initialize the gallery as a LIST with the first embedding
                            customer_database[persistent_id] = [embedding] 
                            entry_times[persistent_id] = current_time
                            reid_match_scores[persistent_id] = 1.0 
                            next_customer_id += 1
                    else:
                        hud_text = "Crop Error"

            # 2. EXISTING CUSTOMER
            elif yolo_id in id_map:
                p_id = id_map[yolo_id]
                
                # --- FIXED: CENTER FRAME RULE ---
                margin_x = int(frame_width * 0.10) 
                margin_y = int(frame_height * 0.10) 
                is_center_frame = (bbox[0] > margin_x and bbox[1] > margin_y and 
                                   bbox[2] < (frame_width - margin_x) and bbox[3] < (frame_height - margin_y))
                
                margin = 50
                is_fully_in_frame = (bbox[0] > margin and bbox[1] > margin and bbox[2] < (frame_width - margin) and bbox[3] < (frame_height - margin))
                
                # We only update the gallery if they are fully in frame, not occluded, AND in the center
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

                # --- ALWAYS GENERATE A LABEL ---
                time_in_frame = current_time - entry_times.get(p_id, current_time)
                reid_score = reid_match_scores.get(p_id, 1.0)
                occ_marker = "*" if is_occluded else ""
                
                gallery_size = len(customer_database.get(p_id, []))
                
                # I added a little "(EDGE)" marker to the HUD so you can visually see when the rule is active!
                edge_marker = "" if is_center_frame else " (EDGE)"
                
                hud_text = f"{p_id}{occ_marker}{edge_marker} | Angles: {gallery_size} | YOLO:{yolo_conf:.2f} | ReID:{reid_score:.2f}"
                
            labels.append(hud_text)

    if SHOW_LIVE_VIDEO:
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
        if labels:
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # --- RESIZE DISPLAY WINDOW ---
        # Multiply by 0.5 to cut the size in half. Change to 0.75 or 0.3 depending on your screen size!
        scale_factor = 0.5 
        display_width = int(frame_width * scale_factor)
        display_height = int(frame_height * scale_factor)
        display_frame = cv2.resize(annotated_frame, (display_width, display_height))

        # Show the newly resized frame instead of the massive original one
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
print(f"👥 Total Profiles Created: {len(customer_database)} (Target: 2)")
print(f"🎚️  Match Threshold:  {SIMILARITY_THRESHOLD}")
print(f"🖼️  Gallery Limit:    {MAX_GALLERY_SIZE} angles per person")
print("="*40 + "\n")

# Print out exactly how many angles each profile learned
print("📸 Profile Gallery Details:")
for pid, gallery in customer_database.items():
    print(f"   - {pid}: Captured {len(gallery)} distinct angles.")

cap.release()
cv2.destroyAllWindows()