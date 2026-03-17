# 🚀 Object Detection and Monocular Depth Estimation using YOLOv8

## 📌 Overview
This project implements a complete computer vision pipeline for **vehicle detection and real-world distance estimation** using monocular images.

The system is developed and evaluated using the **KITTI Vision Benchmark Suite**, a widely used dataset in autonomous driving research, ensuring realistic road-scene evaluation.

It combines **YOLOv8-based object detection**, **IoU-based evaluation**, and **camera intrinsic calibration** to estimate vehicle distances from the camera.

---

## 🎯 Key Features
- 🔍 Vehicle detection using **YOLOv8x**
- 📏 IoU-based filtering of predictions (threshold = 0.6)
- 📊 Performance evaluation using **Precision & Recall**
- 📐 Monocular depth estimation using **camera intrinsic parameters**
- 📍 Real-world distance calculation from image coordinates
- 🚘 Evaluated on real-world autonomous driving data from **KITTI dataset**

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Libraries:** OpenCV, NumPy, Matplotlib  
- **Model:** YOLOv8 (Ultralytics)  
- **Dataset:** KITTI Vision Benchmark Suite  
- **Concepts:** IoU, Precision/Recall, Camera Calibration, 3D Projection  

---

## ⚙️ Methodology

### 1. Object Detection
- Used pretrained **YOLOv8x model**
- Detected vehicles in road-scene images

### 2. IoU Filtering
- Compared predictions with ground truth
- Applied **IoU threshold = 0.6**
- Removed false positives

### 3. Performance Evaluation
- Computed:
  - Precision
  - Recall

### 4. Depth Estimation
- Used **camera intrinsic matrix**
- Converted image coordinates → real-world coordinates
- Estimated distance using Euclidean geometry

---

## 📊 Results
- Accurate detection for nearby vehicles  
- Distance estimation follows ground-truth trend for close objects  
- Performance degrades for:
  - Distant objects  
  - Overlapping vehicles  
  - Curved roads  
  - Complex backgrounds  

---

## ⚠️ Limitations
- Monocular depth estimation has limited accuracy  
- Sensitive to bounding box alignment  
- Errors increase with distance  
- Challenging scenarios:
  - Occlusions
  - Road curvature
  - Small/distant objects  

---

## 📂 Project Structure

```bash
object-detection-depth-estimation/
├── README.md
├── requirements.txt
├── .gitignore
│
├── report/
│   └── Task_Report.pdf
│
├── src/
│   ├── detect_objects.py
│   ├── iou_evaluation.py
│   ├── precision_recall.py
│  
│  
│
├── results/
│   ├── images/
│   └── plots/
│
└── sample_data/


---
Future Improvements

Use stereo vision for improved depth accuracy

Integrate LiDAR or sensor fusion

Improve bounding box refinement

Real-time deployment on embedded systems

Add object tracking (e.g., Kalman Filter, SORT)

🧠 Learning Outcomes

Applied computer vision techniques on a real-world autonomous driving dataset (KITTI)

Developed a complete object detection and evaluation pipeline

Implemented IoU-based filtering and performance metrics

Applied camera calibration for real-world distance estimation

👨‍💻 Author

Rajveersinh Suratiya
Master’s Student – Electrical Eng And Embedded Systems
Ravensburg-Weingarten University (RWU), Germany

## ▶️ How to Run

### 1. Clone Repository
```bash
git clone https://github.com/RajveersinhS/Object-Detection-And-Depth-Estimation-Project.git
