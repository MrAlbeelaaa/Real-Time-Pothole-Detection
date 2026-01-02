# 🚧 Real-Time Pothole Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)

A computer vision application designed to improve road safety by detecting potholes in real-time from dashcam footage. This project uses **YOLOv8** for object detection and **Streamlit** for the interactive dashboard.

## 🌟 Key Features
* **Real-Time Detection:** Processes video feeds at high FPS using YOLOv8.
* **Data Logging:** Automatically records pothole counts and timestamps to a CSV file for analysis.
* **Confidence Control:** Adjustable threshold slider to fine-tune detection sensitivity.
* **Visual Alerts:** On-screen warnings when hazardous road conditions are detected.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Computer Vision:** OpenCV, Ultralytics YOLOv8
* **Interface:** Streamlit
* **Data Handling:** Pandas, NumPy

## 🚀 How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MrAlbeelaaa/Real-Time-Pothole-Detection.git](https://github.com/MrAlbeelaaa/Real-Time-Pothole-Detection.git)
    cd Real-Time-Pothole-Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 📊 Project Output
* **Input:** Dashcam video (MP4/AVI)
* **Output:** Annotated video stream + `pothole_log.csv` containing detection statistics.

---
*Developed by Amrit Raj as a Major Project | January 2026*