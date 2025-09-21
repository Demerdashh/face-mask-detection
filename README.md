# 😷 Real-Time Face Mask Detection

A practical computer vision project that detects whether a person is **wearing a mask or not** in real time using **PyTorch** and **OpenCV**.  
The system runs directly on laptop i7 CPU and includes a **color-coded FPS counter** to prove real-time performance.

---

# Table of Contents

1. [🚀 Why This Matters](#why-this-matters)  
2. [🏗️ System Architecture](#system-architecture)
3. [📁 Project Structure](#project-structure)
4. [📊 Performance Summary](#performance-summary)  
5. [🔧 How It Works](#how-it-works)  
6. [🖼️ Example Results](#example-results)  
7. [🛠️ Requirements](#requirements)  
8. [👤 Author](#author)  

---

## 🚀 Why This Matters <a name="why-this-matters"></a>
Face mask detection is a widely used **classification + deployment** task in computer vision.  
This project demonstrates:
- Real-time inference with **PyTorch CNN (MobileNetV3-Small, transfer learning)**.  
- Integration of **OpenCV Haar cascades** for face ROI extraction.  
- Live **FPS measurement** to evaluate system viability on constrained hardware.  

This project was completed during an **8-week internship at ARCH Technologies** as part of the “Sharpening Your Hidden Skills for a Brighter Future” program.

---

## 🏗️ System Architecture <a name="system-architecture"></a>
```mermaid
flowchart LR
    A[Webcam Input] --> B[Face Detection — Haar Cascade]
    B --> C[Crop & Preprocess Face ROI]
    C --> D[PyTorch CNN Model — MobileNetV3-Small]
    D --> E[Mask / No-Mask Prediction + Confidence]
    E --> F[Draw Label + Color-Coded FPS Counter]
    F --> G[Display Output Window]
```

---

## 📁 Project Structure <a name ="project-structure"></a>
<img width="857" height="336" alt="Screenshot 2025-09-22 004756" src="https://github.com/user-attachments/assets/a7b10a23-4a91-4650-94bd-326e0c9019f0" />

---

## 📊 Performance Summary <a name="performance-summary"></a>

| Metric                     | Value                          |
|----------------------------|--------------------------------|
| Average FPS (Laptop CPU)   | 11.3 / 10.9                    |
| Resolution                 | 640×480                        |
| Model                      | MobileNetV3-Small (transfer learning) |
| Latency per frame          | ~90 ms                         |
| Detection Accuracy         | ✅ Reliable on test dataset (masked vs unmasked faces) |

- 🟢 >15 FPS = Real-time viable (Jetson / GPU)  
- 🟡 10–15 FPS = Acceptable on CPU (this project’s range)  
- 🔴 <10 FPS = Too slow  

---

## 🔧 How It Works <a name="how-it-works"></a>

### 👁️ Face Detection
- Uses **OpenCV’s Haar Cascade** to detect faces in each frame.
- Crops **Regions of Interest (ROIs)** for classification.

### 🧹 Preprocessing
- Resize face ROIs to **224×224**.
- Normalize using **ImageNet statistics** (mean & std).

### 🧠 Model Training
- Transfer learning with **MobileNetV3-Small** in PyTorch.
- Fine-tuned last layers for **binary classification**: `Mask` vs `No Mask`.
- **Metadata** (normalization stats, labels) saved alongside model.

### 🚀 Real-Time Inference
- Predictions drawn as **bounding boxes** with class + confidence.
- FPS measured as `1 / (current_time - prev_time)`.
- **FPS counter colors**:
  - 🟢 >15 = Green (real-time ready)
  - 🟡 10–15 = Yellow (CPU-bound, acceptable)
  - 🔴 <10 = Red (not real-time)

---

## 🖼️ Example Results <a name="example-results"></a>

✅ Detects mask vs no-mask in real time  
✅ Displays class label + confidence  
✅ FPS counter (yellow for ~11 FPS on CPU)  
✅ Works with multiple faces in frame  

![mask detection](https://github.com/user-attachments/assets/957a3823-a429-4739-8602-190cb54cc1af)

---

## 🛠️ Requirements <a name="requirements"></a>

- Python 3.8+
- PyTorch
- sklearn (`train_test_split only`)
- Torchvision
- OpenCV (`opencv-python`)
- NumPy

---

## 👤 Author <a name="author"></a>
Built with ❤️ by Youssef Ahmed El Demerdash
During the ARCH Technologies Internship (2025)
