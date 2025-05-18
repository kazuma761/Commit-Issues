import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import tensorflow as tf
# Configure TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')
# Enable memory growth
physical_devices = tf.config.list_physical_devices('CPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from waste_type_model import load_waste_type_model, predict_waste_type
from waste_detection_model import load_waste_detection_model, detect_waste

st.title("Waste Type Video Classifier")
st.write("Upload a video to detect and categorize waste (wood, glass, etc.)")

# File uploader allows user to add their own video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Display the uploaded video
    st.video(video_path)
    st.write("Processing video, please wait...")

    try:
        # Load models
        waste_type_model = load_waste_type_model()
        waste_detection_model = load_waste_detection_model()

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Initialize variables to store results
        frame_results = []
        
        # Process every nth frame (e.g., every 30th frame)
        frame_interval = 30
        frame_count = 0
        progress_bar = st.progress(0)
        
        # Get total frames for progress calculation
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Update progress bar
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            
            if frame_count % frame_interval != 0:
                continue
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect waste in frame
            detected, waste_roi = detect_waste(frame_rgb, waste_detection_model)
            
            if detected:
                # Predict waste type
                waste_type, confidence = predict_waste_type(waste_roi, waste_type_model)
                frame_results.append(waste_type)
                
        cap.release()
        progress_bar.empty()
        
        if frame_results:
            # Count occurrences of each waste type
            from collections import Counter
            waste_counts = Counter(frame_results)
            
            # Display results
            st.success("Video Analysis Complete!")
            st.write("### Detected Waste Types:")
            
            # Create a bar chart of waste type frequencies
            import plotly.express as px
            df = pd.DataFrame(list(waste_counts.items()), columns=['Waste Type', 'Count'])
            fig = px.bar(df, x='Waste Type', y='Count', title='Detected Waste Types in Video')
            st.plotly_chart(fig)
            
            # Show most common waste type
            most_common = waste_counts.most_common(1)[0]
            st.write(f"Most common waste type: **{most_common[0]}** (detected {most_common[1]} times)")
        else:
            st.warning("No waste detected in the video.")

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

st.markdown("---")
st.markdown("""
### Supported waste categories:
- Glass
- Metal
- Paper
- Plastic
- Wood
- Other
""")
