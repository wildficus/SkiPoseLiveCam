import json
import os
import numpy as np
import cv2
from ultralytics import YOLO

# Function to process each JSON file
def process_json_file(file_path, base_directory, model, all_distances, correct_keypoints_total, total_keypoints_total):
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

                    # Calculate accuracy metrics
                    keypoint_pairs = [
                        (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
                        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)
                    ]
                    distances = []
                    correct_keypoints = 0
                    total_keypoints = len(keypoint_pairs)

                    for label_idx, detect_idx in keypoint_pairs:
                        if label_idx < len(keypoints) and detect_idx < len(keypoints_norm):
                            lx, ly, lv = keypoints[label_idx]
                            if lv > 0:  # Only consider labeled keypoints with visibility > 0
                                dx, dy = keypoints_norm[detect_idx]
                                lx /= width
                                ly /= height
                                distance = np.sqrt((lx - dx) ** 2 + (ly - dy) ** 2)
                                distances.append(distance)
                                all_distances.append(distance)
                                if distance <= 0.1:  # Example threshold for PCK
                                    correct_keypoints += 1
                    
                    correct_keypoints_total.append(correct_keypoints)
                    total_keypoints_total.append(total_keypoints)
    return all_distances, correct_keypoints_total, total_keypoints_total

# Function to extract results from model predictions
def extract_results(kpts):
    for result in kpts:
        keypoints = result.keypoints  # Keypoints object for pose outputs    
    keypoints_pixel = keypoints.data.numpy()[0]
    keypoints_pixel = np.delete(keypoints_pixel, 2, 1)  
    keypoints_norm = keypoints.xyn.numpy()[0] 
    return [keypoints_pixel, keypoints_norm]

# Base directory containing the annotations and images
base_directory = 'C:/Users/maria/Desktop/Master/Dizertatie/injuryski_dataset/injuryski_dataset'
annotations_directory = os.path.join(base_directory, 'annotations')

# Initialize the model
model = YOLO('yolov8l-pose.pt')

# List of JSON files to process in the annotations directory
json_files = [f'injuryski_{i}.json' for i in range(1000, 1006)]

# Initialize lists to store distances and counts for metrics
all_distances = []
correct_keypoints_total = []
total_keypoints_total = []

# Process each JSON file
for json_file in json_files:
    file_path = os.path.join(annotations_directory, json_file)
    if os.path.exists(file_path):
        all_distances, correct_keypoints_total, total_keypoints_total = process_json_file(
            file_path, base_directory, model, all_distances, correct_keypoints_total, total_keypoints_total)
    else:
        print(f"File {json_file} does not exist in the annotations directory.")

# Calculate and print overall metrics
if all_distances:
    mse = np.mean(np.square(all_distances))
    print(f"MSE (Normalized): {mse}")

    pck_threshold = 0.1  # Adjust as needed
    pck = np.sum(np.array(all_distances) <= pck_threshold) / len(all_distances)
    print(f"PCK@{pck_threshold}: {pck * 100:.2f}%")

    correct_keypoints_total = np.sum(correct_keypoints_total)
    total_keypoints_total = np.sum(total_keypoints_total)
    mAP = correct_keypoints_total / total_keypoints_total if total_keypoints_total > 0 else 0
    print(f"mAP: {mAP * 100:.2f}%")
