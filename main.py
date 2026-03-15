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

# [COMMENTED OUT FOR MP4 TESTING]
# camera_index = 1
# cap = cv2.VideoCapture(camera_index)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# [ADDED FOR MP4 TESTING]
# Place your video file in the same folder as this script, or provide the full path.
video_path = "walking_hallway.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ ERROR: Could not open video file at '{video_path}'. Please check the path and filename.")
    exit()

cv2.namedWindow("Store Tracking POC", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Store Tracking POC", 1280, 720) 

# --- 2. SETUP AI MODELS ---
model = YOLO("yolov8n.pt") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for ReID: {device}")

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
prev_frame_time = 0

customer_database = {} 
id_map = {} 
next_customer_id = 1
reid_match_scores = {} 

# --- 4. MAIN LOOP PARAMETERS ---
SIMILARITY_THRESHOLD = 0.70 
ALPHA = 0.1 
MIN_BOX_AREA = 8000 
MAX_OVERLAP = 0.10  

# --- 5. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    # If the video ends, break the loop naturally
    if not ret: 
        print("End of video reached.")
        break
    
    frame_height, frame_width = frame.shape[:2]

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time

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
                        
                        for known_id, known_embedding in customer_database.items():
                            similarity = F.cosine_similarity(embedding.unsqueeze(0), known_embedding.unsqueeze(0)).item()
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match_id = known_id

                        if best_similarity > SIMILARITY_THRESHOLD:
                            id_map[yolo_id] = best_match_id
                            reid_match_scores[best_match_id] = best_similarity
                            print(f"✅ SUCCESS: Re-linked to {best_match_id} (Score: {best_similarity:.2f})")
                        else:
                            persistent_id = f"CUST-{next_customer_id}"
                            id_map[yolo_id] = persistent_id
                            customer_database[persistent_id] = embedding
                            entry_times[persistent_id] = current_time
                            reid_match_scores[persistent_id] = 1.0 
                            next_customer_id += 1
                            print(f"🆕 CREATED: New profile {persistent_id}")
                    else:
                        hud_text = "Crop Error"

            # 2. EXISTING CUSTOMER
            # [FIX APPLIED HERE: Changed 'if' to 'elif' to prevent double inference on the first frame]
            elif yolo_id in id_map:
                p_id = id_map[yolo_id]
                
                margin = 50
                is_fully_in_frame = (bbox[0] > margin and bbox[1] > margin and bbox[2] < (frame_width - margin) and bbox[3] < (frame_height - margin))
                
                if is_fully_in_frame and not is_occluded:
                    current_embedding = get_embedding(frame, bbox)
                    if current_embedding is not None:
                        old_embedding = customer_database[p_id]
                        updated_embedding = (1.0 - ALPHA) * old_embedding + (ALPHA) * current_embedding
                        customer_database[p_id] = F.normalize(updated_embedding, p=2, dim=0)

                time_in_frame = current_time - entry_times.get(p_id, current_time)
                reid_score = reid_match_scores.get(p_id, 1.0)
                
                occ_marker = "*" if is_occluded else ""
                hud_text = f"{p_id}{occ_marker} | {time_in_frame:.0f}s | YOLO:{yolo_conf:.2f} | ReID:{reid_score:.2f}"
                
            labels.append(hud_text)

    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    if labels:
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

    cv2.putText(annotated_frame, f"RES: {frame_width}x{frame_height} | FPS: {int(fps)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Store Tracking POC", annotated_frame)

    # Press 'q' to quit early, or spacebar to pause the video
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord(' '): 
        cv2.waitKey(0) # Pauses until you press spacebar again

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()