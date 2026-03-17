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

# Function to calculate Rc and Pc based on the formulas
def calculate_rc_pc(tp_c, fn_c, fp_c):
    # Rc = TPc / (TPc + FN(c))
    Rc = tp_c / (tp_c + fn_c) if (tp_c + fn_c) != 0 else 0
    # Pc = TPc / (TPc + FP(c))
    Pc = tp_c / (tp_c + fp_c) if (tp_c + fp_c) != 0 else 0
    return Rc, Pc

# Paths to the folders
images_folder = "KITTI_Selection/images"
labels_folder = "KITTI_Selection/labels"
output_folder = "FOR Pecision and recall\presition and recall outputs\Class_outputs"

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

        # Initialize counts for TP, FN, and FP
        tp_c = 0
        fn_c = len(gt_boxes)
        fp_c = 0
        matched_gt_boxes = set()  # To track the matched ground truth boxes

        # Filter boxes based on IoU
        filtered_boxes = []
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            detected_box = [x1, y1, x2, y2]
            
            # Compute IoU for all ground truth boxes
            iou_values = [calculate_iou(detected_box, gt_box) for gt_box in gt_boxes]
            
            # Check if we have any ground truth boxes, and find the best match if available
            if iou_values:
                best_iou = max(iou_values)
                best_gt_box_idx = iou_values.index(best_iou)
            else:
                best_iou = 0
                best_gt_box_idx = -1

            # If IoU is above threshold, count as true positive
            if best_iou >= 0.60:  # IoU threshold
                if best_gt_box_idx not in matched_gt_boxes:
                    tp_c += 1
                    fn_c -= 1  # If TP, reduce FN count
                    matched_gt_boxes.add(best_gt_box_idx)  # Mark ground truth box as matched
                    filtered_boxes.append(detected_box)
                else:
                    fp_c += 1  # If already matched, count as false positive
            else:
                fp_c += 1

        # Calculate Rc and Pc based on TP, FN, FP
        Rc, Pc = calculate_rc_pc(tp_c, fn_c, fp_c)

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

        # Draw a white box on the right side of the image for displaying Rc and Pc
        box_width = 150  # Width of the box on the right side
        box_height = 100  # Height of the box
        right_box_x1 = image.width - box_width - 10
        right_box_y1 = 10
        right_box_x2 = image.width - 10
        right_box_y2 = right_box_y1 + box_height
        draw.rectangle([right_box_x1, right_box_y1, right_box_x2, right_box_y2], fill="white")

        # Display Rc and Pc values inside the white box
        draw.text((right_box_x1 + 10, right_box_y1 + 10), f"Rc: {Rc:.2f}", fill="black", font=font)
        draw.text((right_box_x1 + 10, right_box_y1 + 30), f"Pc: {Pc:.2f}", fill="black", font=font)

        # Save the output image
        output_img_path = os.path.join(output_folder, f"{file_index}_iou.png")
        image.save(output_img_path)

        print(f"Processed and saved: {output_img_path}")
