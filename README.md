# 🛒 Retail Store Customer Tracking & Re-Identification (ReID) POC

An advanced computer vision pipeline designed to track and re-identify customers across a camera's field of view. This Proof of Concept (POC) handles complex scenarios like occlusions, customers leaving and returning to the frame, and multi-angle identity matching.

### 🎥 Demonstration

<div align="center">
<video src="https://github.com/mitch-gres/retail-store-customer-tracking/blob/main/walking_hallway_3_tracked_hud.mp4" width="100%" controls autoplay></video>    Your browser does not support the video tag.
  </video>
</div>

---

## ✨ Key Features

* **Custom Feature Gallery (ReID):** Abandons traditional moving-average filters in favor of a "Feature Gallery." The system saves up to 5 distinct visual angles (front, back, side) of a customer, allowing for highly accurate re-identification even after they leave and re-enter the camera view.
* **Intelligent Identity Capture:** Utilizes "Center Frame" logic to prevent corrupted data. The system only learns new angles when a customer is fully visible in the center of the frame, ignoring partial crops on the edges.
* **Dual-Timeline Tracking:** Separately tracks a customer's total time in the store versus their active time in a specific aisle. 
* **Visit Counting:** Automatically increments a visit counter if a customer leaves the aisle and returns later.
* **Dynamic HUD Alerts:** On-screen display automatically color-codes text (White -> Yellow -> Red) based on how long a customer has lingered in an area or if they are a returning visitor.
* **Accelerated Headless Testing:** Built-in frame-skipping and headless execution modes allow for rapid testing and hyperparameter tuning on pre-recorded MP4s.

## 🧠 How It Works

1.  **Spatial Detection:** Uses `YOLOv8-seg` combined with the `BoT-SORT` tracker to establish robust frame-to-frame bounding boxes and spatial tracking.
2.  **Feature Extraction:** Bounding box crops are fed into a headless `ResNet50` model to extract a 2048-dimensional visual fingerprint of the customer's clothing and appearance.
3.  **Cosine Similarity Matching:** When a new YOLO ID is generated (e.g., after an occlusion), the system compares the new visual fingerprint against the Feature Gallery of all known customers using Cosine Similarity to seamlessly restore their original ID.

## 🛠️ Tech Stack

* [Ultralytics (YOLOv8)](https://github.com/ultralytics/ultralytics) - Instance Segmentation & Spatial Tracking
* [PyTorch & TorchVision](https://pytorch.org/) - ResNet50 Feature Extraction
* [Supervision](https://github.com/roboflow/supervision) - Advanced Bounding Box & HUD Annotation
* [OpenCV](https://opencv.org/) - Video Processing & I/O

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is highly recommended to run this on a CUDA-enabled GPU for real-time processing.

```bash
pip install ultralytics supervision torch torchvision opencv-python numpy
