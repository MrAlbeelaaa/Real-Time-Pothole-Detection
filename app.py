"""
Major Project: Real-Time Pothole Detection System
-------------------------------------------------
Description: A computer vision application using YOLOv8 to detect road hazards
             from dashcam footage. It includes real-time inference, FPS monitoring,
             and automated data logging for research analysis.
             
Author: Amrit Raj
Date: January 2026
Tech Stack: Python, Streamlit, OpenCV, Ultralytics YOLO, Pandas
"""

# --- Standard Library Imports ---
import os
import time
import tempfile

# --- Third-Party Imports ---
import cv2
import numpy as np
import pandas as pd  # Added for data logging
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="AI Road Safety System", layout="wide")

st.title("🚧 Real-Time Pothole & Accident Detection")
st.sidebar.header("Configuration")

# 1. Load the Model
@st.cache_resource
def load_model():
    # Looks for model.pt in the same folder. 
    # Ensure 'best.pt' is renamed to 'model.pt' or change this string.
    model_path = "model.pt"  # Changing to best.pt as per standard training output
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found! Please check the file name.")
        return None
    return YOLO(model_path)

try:
    with st.spinner("Loading AI Model..."):
        model = load_model()
        if model:
            st.sidebar.success("✅ AI Model Loaded Successfully")
        else:
            st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---------------------------------------------------------
# INTERFACE LAYOUT
# ---------------------------------------------------------
source_type = st.sidebar.radio("Select Input Source", ["Upload Video", "Webcam"])
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, help="Minimum probability to detect a pothole.")

# Create two columns for video feed
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Feed")
    frame_placeholder1 = st.empty()

with col2:
    st.subheader("AI Detection Output")
    frame_placeholder2 = st.empty()

# Dashboard Metrics (KPIs)
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    kpi_fps = st.empty()
with metric_col2:
    kpi_count = st.empty()
with metric_col3:
    kpi_status = st.empty()

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def save_log(timestamp, count, conf):
    """
    Logs detection data to a CSV file for research analysis.
    """
    file_exists = os.path.isfile("pothole_log.csv")
    
    # Create a simple dataframe for the current frame
    df = pd.DataFrame([[timestamp, count, conf]], columns=["Timestamp", "Pothole_Count", "Avg_Confidence"])
    
    # Append to CSV (create header if file doesn't exist)
    df.to_csv("pothole_log.csv", mode='a', header=not file_exists, index=False)

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
if source_type == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a dashcam video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Create a temp file to read the video safely
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_stop = st.sidebar.button("Stop Processing")
        
        while cap.isOpened() and not st_stop:
            success, frame = cap.read()
            if not success:
                st.info("Video Ended")
                break
                
            start_time = time.time()
            
            # STEP 1: PRE-PROCESSING
            # Resize frame to standard YOLO size (640x640) for higher FPS
            # We keep the aspect ratio in mind, but for raw speed 640x640 is standard
            frame_resized = cv2.resize(frame, (640, 640))
            
            # STEP 2: INFERENCE
            results = model.predict(frame_resized, conf=confidence)
            
            # STEP 3: VISUALIZATION
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            # Count Hazards & Get Confidence
            boxes = results[0].boxes
            pothole_count = len(boxes)
            avg_conf = 0
            if pothole_count > 0:
                # Calculate average confidence of all detected potholes
                avg_conf = float(boxes.conf.mean().cpu().numpy())
                
                # LOGGING: Save data for research paper
                current_time = time.strftime("%H:%M:%S")
                save_log(current_time, pothole_count, avg_conf)

            # Update Metrics
            kpi_fps.metric("Processing Speed", f"{fps:.1f} FPS")
            kpi_count.metric("Hazards Detected", f"{pothole_count}")
            
            if pothole_count > 0:
                kpi_status.error(f"⚠️ CAUTION: {pothole_count} HAZARDS AHEAD")
            else:
                kpi_status.success("✅ ROAD CLEAR")
            
            # Display Side-by-Side (Convert BGR to RGB for Streamlit)
            # Resize original frame to match annotated frame size for clean UI
            frame_display = cv2.resize(frame, (640, 640))
            
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            frame_placeholder1.image(frame_rgb, channels="RGB", use_container_width=True)
            frame_placeholder2.image(annotated_rgb, channels="RGB", use_container_width=True)
            
        cap.release()

elif source_type == "Webcam":
    run = st.sidebar.checkbox('Start Webcam')
    
    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not access webcam. Please check your camera settings.")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Frame capture failed.")
                    break
                    
                start_time = time.time()
                
                # Resize for consistency
                frame_resized = cv2.resize(frame, (640, 640))
                
                # Inference
                results = model.predict(frame_resized, conf=confidence)
                annotated_frame = results[0].plot()
                
                # FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                
                # Metrics
                pothole_count = len(results[0].boxes)
                kpi_fps.metric("Speed", f"{fps:.1f} FPS")
                kpi_count.metric("Hazards", f"{pothole_count}")
                
                # Show Output
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder2.image(annotated_rgb, channels="RGB", use_container_width=True)
                
            cap.release()