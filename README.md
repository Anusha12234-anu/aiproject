 AI Mask Detection System:-
📌 Project Overview

This project is an AI-powered mask detection system built with YOLOv8, Streamlit, and OpenCV. It can detect whether people are wearing masks or not from both uploaded images and real-time webcam feed. The system is designed as a health-safety monitoring tool that can be extended to public spaces, workplaces, or healthcare environments.

 Features
* YOLOv8-based object detection for high-accuracy mask detection
* Streamlit Web App – Upload multiple images and view detection results with bounding boxes
* Real-time Detection – Uses OpenCV to run live webcam-based mask detection
* User-friendly Interface – Simple design for easy testing and deployment
* Scalable Solution – Can be extended to large-scale safety compliance monitoring

=> Tech Stack

Python
YOLOv8 (Ultralytics) – Object detection model
PyTorch – Model inference
Streamlit – Web app interface
OpenCV – Real-time webcam integration
PIL – Image processing

📂 Project Structure
├── app.py                  # Streamlit app for image-based detection
├── livecamera.py           # Real-time detection using webcam
├── project.py              # Test script for inference on sample images
├── mask_withoutmask_model.pt # Trained YOLOv8 model
├── uploads/                # Uploaded images folder
├── runs/                   # YOLO detection results

=> How to Run

1️⃣ Clone the repository
git clone https://github.com/Anusha12234-anu/aiproject.git
cd aiproject
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run Streamlit app (image detection)
streamlit run app.py
4️⃣ Run real-time webcam detection
python livecamera.py



🎯 Future Improvements
🔹 Add multi-class detection (e.g., mask, no mask, incorrect mask)
🔹 Deploy on cloud (AWS/GCP) for large-scale use
🔹 Integrate alert system for real-time violations

✨ Developed with passion for AI in healthcare & safety
