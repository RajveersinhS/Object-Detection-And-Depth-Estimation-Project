import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Define font settings
font_path = "arial.ttf"  # Path to the Arial font file
font_size = 11
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

# Function to convert pixel coordinates to 3D world coordinates
def convert_to_3D(pixel_coordinate, K_inv, scalar):
    pixel_homogeneous = np.append(pixel_coordinate, 1).reshape(3, 1)
    ray_direction = np.dot(K_inv, pixel_homogeneous)
    point_3D = scalar * ray_direction.flatten()
    return point_3D

# Function to calculate real-world distance using 3D conversion
def calculate_real_world_distance_3D_conversion(bottom_center, K, camera_height):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K_inv = np.linalg.inv(K)
    real_world_Z = camera_height * fy / (bottom_center[1] - cy)
    bottom_center_3D = convert_to_3D(bottom_center, K_inv, real_world_Z)
    distance = np.linalg.norm(bottom_center_3D)
    return distance

# List of file indices
file_indices = ['006037', '006042', '006048', '006054', '006059', '006067', '006097', '006098', '006206', '006211', '006227', '006253', '006291', '006310', '006312', '006315', '006329', '006374']

# Create YOLO model (consider pre-trained models with higher accuracy)
model = YOLO("yolov8x.pt")  # Adjust for your preferred model

# Define camera height
camera_height = 1.65  # meters

for file_index in file_indices:
    # Load the ground truth bounding boxes and distances
    gt_boxes = []
    gt_distances = []
    with open(f"KITTI_Selection/labels/{file_index}.txt", 'r') as f:
        for line in f:
            values = line.strip().split(' ')
            if len(values) == 6:
                _, x1, y1, x2, y2, distance = values
                gt_boxes.append((float(x1), float(y1), float(x2), float(y2)))
                gt_distances.append(float(distance))

    # Path to the image
    img_path = f"KITTI_Selection/images/{file_index}.png"

    # Object detection with adjustments for accuracy
    results = model.predict(
        source=img_path,
        conf=0.25,  # Increase confidence threshold for stricter detection (0.3 to 0.5)
        augment=True,  # Enable data augmentation for robustness (optional)
    )

    # Extract detection results
    boxes = results[0].boxes.xyxy.tolist()
    confidences = results[0].boxes.conf.tolist()

    # Filter out detections based on IoU with ground truth boxes
    filtered_boxes = []
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        detected_box = [x1, y1, x2, y2]
        max_iou = max([calculate_iou(detected_box, gt_box) for gt_box in gt_boxes])
        if max_iou >= 0.45:
            filtered_boxes.append(box)

    # Save filtered bounding box coordinates
    output_file = fr"Codes\bb_coordinates\{file_index}.txt"  # Using raw string literals for file path
    with open(output_file, 'w') as f:
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            f.write(f"{x1} {y1} {x2} {y2}\n")

    # Load calibration matrix
    calib_path = f"KITTI_Selection/calib/{file_index}.txt"
    calib = np.loadtxt(calib_path)  # Load the calibration matrix from file

    # Load bounding box coordinates from YOLO file
    bb_coords = np.loadtxt(output_file, usecols=(0, 1, 2, 3), dtype='float')

    # Ensure bb_coords is 2D
    if bb_coords.ndim == 1:
        bb_coords = bb_coords.reshape(-1, 4)

    # Calculate distances for each YOLO bounding box using 3D conversion method
    distances_3D_conversion = []
    for bbox in bb_coords:
        x1, y1, x2, y2 = bbox
        bottom_center = [(x1 + x2) / 2, y2]  # Calculate bottom center of bounding box
        distance = calculate_real_world_distance_3D_conversion(bottom_center, calib, camera_height)
        distances_3D_conversion.append(distance)

    # Save distances in the 5th column of the YOLO bounding box files
    with open(output_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line, distance in zip(lines, distances_3D_conversion):
            line = line.strip()  # Remove trailing newline
            f.write(f"{line} {distance:.2f}\n")  # Append the distance in the 5th column

    # Load the image
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)

    # Draw YOLO bounding boxes
    for bbox in bb_coords:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Draw ground truth bounding boxes, white box, and display distances
    for gt_box, gt_distance, yolo_distance in zip(gt_boxes, gt_distances, distances_3D_conversion):
        x1, y1, x2, y2 = gt_box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        # Create a white filled rectangle at the top-left corner of the GT bounding box
        text_box_start_x, text_box_start_y = x1, y1 - 30
        text_box_end_x, text_box_end_y = text_box_start_x + 100, text_box_start_y + 30
        draw.rectangle([(text_box_start_x, text_box_start_y), (text_box_end_x, text_box_end_y)], fill=(255, 255, 255))
        # Display YOLO and GT distances in the white box
        draw.text((text_box_start_x + 5, text_box_start_y), f"YOLO: {yolo_distance:.2f} m", fill="black", font=font)
        draw.text((text_box_start_x + 5, text_box_start_y + 15), f"GT: {gt_distance:.2f} m", fill="black", font=font)

    # Save the image with bounding boxes and distances
    output_img_path = fr"Codes\Output_images\{file_index}.png"  # Using raw string literals for file path
    image.save(output_img_path)

    # Optionally display the image (uncomment the next line to display images)
    # image.show()

# Initialize lists to store distances for plotting
all_gt_distances = []
all_yolo_distances = []

# Iterate over each file index
for idx in file_indices:
    # Load YOLO distances
    bb_path = fr"Codes\bb_coordinates\{idx}.txt"  # Using raw string literals for file path
    bb_coords_with_distances = np.loadtxt(bb_path, usecols=(4,), dtype='float', ndmin=1)

    # Load ground truth distances
    gt_path = f"KITTI_Selection/labels/{idx}.txt"
    gt_coords = np.loadtxt(gt_path, usecols=(5,), dtype='float', ndmin=1)

    # Ensure both arrays have the same size
    if len(bb_coords_with_distances) == len(gt_coords):
        # Append distances for plotting
        all_yolo_distances.extend(bb_coords_with_distances)
        all_gt_distances.extend(gt_coords)
    else:
        print(f"Warning: Mismatched sizes in file {idx}")

# Convert to numpy arrays for plotting
all_gt_distances = np.array(all_gt_distances)
all_yolo_distances = np.array(all_yolo_distances)

# Plot ground truth vs YOLO distances
plt.figure(figsize=(10, 6))
plt.scatter(all_yolo_distances, all_gt_distances, color='blue', label='YOLO vs GT')
plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Distance calculated using camera information (m)')
plt.ylabel('Distance provided in ground truth (m)')
plt.title('Distance estimation compared to the ground truth')
plt.xticks(np.arange(0, 101, 20))
plt.yticks(np.arange(0, 101, 20))
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.show()
