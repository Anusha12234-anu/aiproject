from ultralytics import YOLO

# Load a model
model = YOLO("mask_withoutmask_model.pt")  # trained model file

# Run batched inference on a list of images
results = model(["img.png","img_1.png"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
