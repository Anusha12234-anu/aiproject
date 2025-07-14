import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import shutil

# Setup
st.set_page_config(page_title="Mask Detection", layout="wide")
st.title("😷 YOLOv8 Mask Detection - Multiple Images")
st.markdown("Upload multiple images to detect faces with or without masks using your trained model.")

# Load the YOLOv8 model
model = YOLO("mask_withoutmask_model.pt")  # Make sure it's in the same folder

# Create folders
UPLOAD_DIR = "uploads"
RESULTS_DIR = "runs/detect/predict"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Clear old prediction results
if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)

# File uploader for multiple files
uploaded_files = st.file_uploader("📤 Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("📸 Uploaded Images")
    img_paths = []

    for uploaded_file in uploaded_files:
        image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        img_paths.append(image_path)
        st.image(image_path, width=300, caption=uploaded_file.name)

    with st.spinner("Running YOLOv8 on uploaded images..."):
        results = model(img_paths, save=True, conf=0.3)

    st.success("✅ Detection Complete")

    # Class labels (adjust if your model uses different names)
    class_map = ['mask', 'without_mask']

    st.subheader("📌 Detection Results")

    for idx, result in enumerate(results):
        img_name = os.path.basename(img_paths[idx])
        result_img_path = os.path.join(RESULTS_DIR, img_name)

        st.image(result_img_path, width=400, caption=f"Detected: {img_name}")

        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"➡️ **{class_map[cls_id]}** detected with **{conf*100:.1f}%** confidence")
        else:
            st.warning(f"⚠️ No person detected in **{img_name}**")
