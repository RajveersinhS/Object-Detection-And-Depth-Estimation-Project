import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Define font settings
font_path = "arial.ttf"  # Adjust font path as needed
font_size = 12
font = ImageFont.truetype(font_path, font_size)

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area
    return iou

# Paths to the folders
images_folder = "KITTI_Selection/images"
labels_folder = "KITTI_Selection/labels"
output_folder = "Codes_for_IOU"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO("./KITTI_Selection/yolov8x.pt")  # Adjust to your YOLO model path

# Process each image in the folder
for image_file in os.listdir(images_folder):
    if image_file.endswith(".png"):  # Process only PNG files
        file_index = os.path.splitext(image_file)[0]
        img_path = os.path.join(images_folder, image_file)
        gt_path = os.path.join(labels_folder, f"{file_index}.txt")

        # Load ground truth bounding boxes
        gt_boxes = []
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    values = line.strip().split(' ')
                    if len(values) == 6:
                        _, x1, y1, x2, y2, _ = values
                        gt_boxes.append((float(x1), float(y1), float(x2), float(y2)))

        # Predict bounding boxes using YOLO model
        results = model.predict(source=img_path, conf=0.4, augment=True)
        boxes = results[0].boxes.xyxy.tolist()
        confidences = results[0].boxes.conf.tolist()

        # Filter boxes based on IoU
        filtered_boxes = []
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            detected_box = [x1, y1, x2, y2]
            max_iou = max([calculate_iou(detected_box, gt_box) for gt_box in gt_boxes], default=0)
            if max_iou >= 0.45:  # IoU threshold
                filtered_boxes.append(detected_box)

        # Load the image
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)

        # Draw ground truth boxes
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = gt_box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        # Draw filtered boxes and IoU values
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            best_iou = max([calculate_iou(box, gt_box) for gt_box in gt_boxes], default=0)
            draw.text((x1, y1 - 15), f"IoU: {best_iou:.2f}", fill="red", font=font)

        # Save the output image
        output_img_path = os.path.join(output_folder, f"{file_index}_iou.png")
        image.save(output_img_path)

        print(f"Processed and saved: {output_img_path}")
