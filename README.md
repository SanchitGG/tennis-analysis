# ​ Tennis Analysis

A computer vision project analyzing tennis match videos to measure player speed, ball shot speed, and shot count. It detects players and tennis balls using YOLO and leverages CNN models to extract court keypoints—an excellent hands-on tool for mastering machine learning and computer vision. :contentReference[oaicite:0]{index=0}

---

##  Demo Output

![Sample Output Video Screenshot]()

*(Insert a screenshot from an output video here for illustration.)* :contentReference[oaicite:1]{index=1}

---

##  Models Used

- **YOLOv8** for detecting tennis players  
- **Fine-tuned YOLO (YOLOv5)** for tennis ball detection  
- CNN-based model for tennis court keypoint extraction  
- **Trained model links**:  
  - Ball detector: [Download link] :contentReference[oaicite:2]{index=2}  
  - Court keypoint model: [Download link] :contentReference[oaicite:3]{index=3}

---

##  Training Notebooks

- `training/tennis_ball_detector_training.ipynb` – YOLO-based ball detection  
- `training/tennis_court_keypoints_training.ipynb` – Court keypoint extraction with PyTorch :contentReference[oaicite:4]{index=4}

---

##  Requirements

- Python 3.8  
- ultralytics  
- pytorch  
- pandas  
- numpy  
- OpenCV :contentReference[oaicite:5]{index=5}

---



"# tennis-analysis" 
