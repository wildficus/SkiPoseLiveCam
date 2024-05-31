import json
import os
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Function to calculate the reference distance
def calculate_reference_distance(keypoints, width, height):
    def distance(kp1, kp2):
        x1, y1, v1 = kp1
        x2, y2, v2 = kp2
        if v1 > 0 and v2 > 0:
            x1 /= width
            y1 /= height
            x2 /= width
            y2 /= height
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return None
    
    neck_head_distance = distance(keypoints[0], keypoints[1])
    if neck_head_distance is not None:
        return neck_head_distance * 0.5

    hip_distance = distance(keypoints[10], keypoints[11])
    if hip_distance is not None:
        return hip_distance * 0.5

    return None

# Function to process each JSON file
def process_json_file(file_path, base_directory, model, results_per_image):
    with open(file_path, 'r') as file:
        data = json.load(file)

    for image in data['images']:
        if image['is_labeled']:
            # Find the corresponding annotation by image id
            image_id = image['frame_id']
            annotation = next((ann for ann in data['annotations'] if ann['image_id'] == image_id), None)
            if annotation:
                keypoints = annotation.get('keypoints', [])
                
                # Load the image
                img_path = os.path.join(base_directory, image['file_name'])
                img = cv2.imread(img_path)
                
                if img is not None:
                    height, width, _ = img.shape

                    # Predict with the model on the frame
                    results = model(img)
                    keypoints_pixel, keypoints_norm = extract_results(results)

                    if len(keypoints_norm) == 0:
                        print(f"No keypoints detected for image {image_id}")
                        continue

                    # Calculate reference distance
                    reference_distance = calculate_reference_distance(keypoints, width, height)
                    if reference_distance is None:
                        print(f"Reference distance could not be calculated for image {image_id}")
                        continue

                    # Initialize results_per_image[image_id] as a dictionary if not already present
                    if image_id not in results_per_image:
                        results_per_image[image_id] = {}

                    # Calculate accuracy metrics
                    keypoint_pairs = [
                        (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
                        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)
                    ]

                    percentages = np.arange(90, 0, -10)
                    thresholds = {percentage: threshold for percentage, threshold in zip(percentages, np.linspace(0.9, 0.1, 9) * reference_distance)}

                    for percentage, threshold in thresholds.items():
                        tp, fp, fn = 0, 0, 0

                        for label_idx, detect_idx in keypoint_pairs:
                            if label_idx >= len(keypoints) or detect_idx >= len(keypoints_norm):
                                continue

                            lx, ly, lv = keypoints[label_idx] 
                            dx, dy = keypoints_norm[detect_idx] 

                            labeled_exists = lx != 0 or ly != 0
                            detected_exists = dx != 0 or dy != 0

                            if labeled_exists and detected_exists:
                                lx /= width
                                ly /= height
                                distance = np.sqrt((lx - dx) ** 2 + (ly - dy) ** 2)
                                if distance <= threshold:
                                    tp += 1
                                else:
                                    fp += 1
                            elif detected_exists:
                                fp += 1
                            elif labeled_exists:
                                fn += 1

                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                        if percentage not in results_per_image[image_id]:
                            results_per_image[image_id][percentage] = {"precisions": [], "recalls": []}

                        results_per_image[image_id][percentage]["precisions"].append(precision)
                        results_per_image[image_id][percentage]["recalls"].append(recall)

    return results_per_image

# Function to extract results from model predictions
def extract_results(kpts):
    keypoints = []
    for result in kpts:
        if result.keypoints is not None:
            keypoints = result.keypoints.data.numpy()[0]
            keypoints_pixel = np.delete(keypoints, 2, 1)
            keypoints_norm = result.keypoints.xyn.numpy()[0]
            return keypoints_pixel, keypoints_norm
    return np.array([]), np.array([])

# Base directory containing the annotations and images
base_directory = 'C:/Users/maria/Desktop/Master/Dizertatie/injuryski_dataset/injuryski_dataset'
annotations_directory = os.path.join(base_directory, 'annotations')

# Initialize the model
model = YOLO('yolov8l-pose.pt')

# List of JSON files to process in the annotations directory
json_files = [f'injuryski_{i}.json' for i in range(1000, 1006)]

# Initialize dictionary to store precision-recall data
results_per_image = {}

# Process each JSON file
for json_file in json_files:
    file_path = os.path.join(annotations_directory, json_file)
    if os.path.exists(file_path):
        results_per_image[json_file] = {}
        results_per_image = process_json_file(file_path, base_directory, model, results_per_image)
    else:
        print(f"File {json_file} does not exist in the annotations directory.")

# Aggregate precision-recall data across all images
precision_recall_data = {percentage: {"precisions": [], "recalls": []} for percentage in np.arange(90, 0, -10)}

for image_results in results_per_image.values():
    for percentage in precision_recall_data:
        if percentage in image_results:
            precision_recall_data[percentage]["precisions"].extend(image_results[percentage]["precisions"])
            precision_recall_data[percentage]["recalls"].extend(image_results[percentage]["recalls"])

# Calculate average precision and recall for each percentage
average_precision_recall_data = {percentage: {"precision": 0, "recall": 0} for percentage in np.arange(90, 0, -10)}

for percentage in precision_recall_data:
    precisions = precision_recall_data[percentage]["precisions"]
    recalls = precision_recall_data[percentage]["recalls"]
    if precisions and recalls:
        average_precision_recall_data[percentage]["precision"] = np.mean(precisions)
        average_precision_recall_data[percentage]["recall"] = np.mean(recalls)

# Plot precision-recall curves
plt.figure(figsize=(12, 6))

for percentage in average_precision_recall_data:
    precision = average_precision_recall_data[percentage]["precision"]
    recall = average_precision_recall_data[percentage]["recall"]
    plt.plot(recall, precision, marker='o', label=f'Threshold {percentage}%')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True)
plt.show()

# Calculate mAP for different ranges
def calculate_map(average_precision_recall_data, start, end):
    precisions = [average_precision_recall_data[percentage]["precision"] for percentage in np.arange(start, end, -10)]
    recalls = [average_precision_recall_data[percentage]["recall"] for percentage in np.arange(start, end, -10)]
    return np.mean([p * r for p, r in zip(precisions, recalls)])

mAP_10_90 = calculate_map(average_precision_recall_data, 90, 10)
mAP_50_90 = calculate_map(average_precision_recall_data, 90, 50)
mAP_90 = average_precision_recall_data[90]["precision"] * average_precision_recall_data[90]["recall"]

print(f"mAP 10-90: {mAP_10_90 * 100:.2f}%")
print(f"mAP 50-90: {mAP_50_90 * 100:.2f}%")
print(f"mAP 90: {mAP_90 * 100:.2f}%")

# Plot AP vs. IoU Threshold
plt.figure()
plt.plot(np.arange(90, 0, -10), [average_precision_recall_data[percentage]["precision"] * average_precision_recall_data[percentage]["recall"] for percentage in np.arange(90, 0, -10)], marker='o')
plt.xlabel('IoU Threshold (%)')
plt.ylabel('Average Precision (AP)')
plt.title('AP vs. IoU Threshold')
plt.grid(True)
plt.show()
