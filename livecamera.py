import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("mask_withoutmask_model.pt")

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Real-time loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Live Inference", annotated_frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
