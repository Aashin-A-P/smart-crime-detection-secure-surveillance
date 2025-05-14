import streamlit as st
import cv2
import tempfile
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import numpy as np

st.set_page_config(page_title="Crime Detection", layout="wide")

# Load model and processor
@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained("./")
    model = ViTForImageClassification.from_pretrained("./")
    return processor, model

processor, model = load_model()

# Initialize crime frame storage
if "crime_frames" not in st.session_state:
    st.session_state["crime_frames"] = []

# Prediction function
def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred_idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[pred_idx]

# Sidebar for input mode
st.sidebar.title("📂 Select Input Source")
app_mode = st.sidebar.radio("Choose Mode", [
    "Upload Video", "IP Webcam", "Real-time CCTV", "Laptop Webcam", "View Crime Frames"
])

st.title("🚨 Real-Time Crime Detection System")

cap = None  # Default

# ---------------- Option 1: Upload Video ----------------
if app_mode == "Upload Video":
    video_file = st.file_uploader("Upload a CCTV video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

# ---------------- Option 2: IP Webcam ----------------
elif app_mode == "IP Webcam":
    ip_url = st.text_input("Enter IP Webcam MJPEG URL", value="http://192.168.43.1:8080/video")
    if st.button("Start Stream") and ip_url:
        cap = cv2.VideoCapture(ip_url)

# ---------------- Option 3: Real-time CCTV ----------------
elif app_mode == "Real-time CCTV":
    cctv_url = st.text_input("Enter CCTV RTSP/HTTP URL")
    if st.button("Start CCTV Stream") and cctv_url:
        cap = cv2.VideoCapture(cctv_url)

# ---------------- Option 4: Laptop Webcam ----------------
elif app_mode == "Laptop Webcam":
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)

# ---------------- Option 5: View Crime Frames ----------------
elif app_mode == "View Crime Frames":
    st.subheader("🖼️ Detected Crime Frames")

    if len(st.session_state["crime_frames"]) == 0:
        st.info("No crime frames stored yet.")
    else:
        for i, frame in enumerate(st.session_state["crime_frames"]):
            st.image(frame, caption=f"Crime Frame #{i+1}", use_column_width=True)

        if st.button("🗑️ Delete All Crime Frames"):
            st.session_state["crime_frames"] = []
            st.success("All stored crime frames have been deleted.")
        
    st.stop()

# ---------------- Video Processing Logic ----------------
if cap and cap.isOpened():
    st.success("✅ Stream started.")
    
    # Containers
    video_display = st.empty()       # For video frame
    prediction_log = st.container()  # For text logs

    predictions = []
    frame_count = 0
    stop_stream = st.button("⛔ Stop Stream")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Show video frame live
        video_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Predict every 10th frame
        if frame_count % 10 == 0:
            label = predict_frame(frame)
            predictions.append((frame_count, label))

            # Save crime frames
            if label.lower() == "crime":
                crime_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.session_state["crime_frames"].append(Image.fromarray(crime_img))

            with prediction_log:
                st.markdown(f"• **Frame {frame_count}: {label}**")

        if stop_stream:
            break

    cap.release()
    st.success("✅ Stream ended.")

    # Final Summary
    st.write("### 📊 Final Summary of Predictions:")
    crime_count = sum(1 for _, l in predictions if l.lower() == "crime")
    normal_count = sum(1 for _, l in predictions if l.lower() == "normal")

    st.write(f"🔴 **Crime frames**: {crime_count}")
    st.write(f"🟢 **Normal frames**: {normal_count}")
    st.write("#### Frame-wise Predictions:")
    for fnum, lbl in predictions:
        st.write(f"• Frame {fnum}: **{lbl}**")

