import json
import os
import numpy as np
import cv2
from ultralytics import YOLO

# Function to process each JSON file
def process_json_file(file_path, base_directory, model):
    with open(file_path, 'r') as file:
        data = json.load(file)

    for image in data['images']:
        if image['is_labeled']:
            print(f"Image: {image['file_name']}, Frame ID: {image['frame_id']}, Video ID: {image['vid_id']}")
            # Find the corresponding annotation by image id
            image_id = image['frame_id']
            annotation = next((ann for ann in data['annotations'] if ann['image_id'] == image_id), None)
            if annotation:
                keypoints = annotation.get('keypoints', [])
                print(f"Keypoints: {keypoints}")
                
                # Load the image
                img_path = os.path.join(base_directory, image['file_name'])
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Draw annotated keypoints
                    for i, keypoint in enumerate(keypoints):
                        x, y, v = keypoint
                        color = (0, 255, 0) if v == 0 else (0, 0, 255)  # Green if v == 0 else Red
                        if x != 0 and y != 0:  # Only draw keypoints that are not (0, 0)
                            cv2.circle(img, (x, y), 5, color, -1)
                            cv2.putText(img, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Predict with the model on the frame
                    results = model(img)
                    keypoints_pixel, _ = extract_results(results)
                    
                    # Draw predicted keypoints
                    for i, (x, y) in enumerate(keypoints_pixel):
                        if x != 0 and y != 0:  # Only draw keypoints that are not (0, 0)
                            color = (255, 0, 0)  # Blue for predicted keypoints
                            cv2.circle(img, (int(x), int(y)), 5, color, -1)
                            cv2.putText(img, str(i), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Calculate accuracy metrics
                    keypoint_pairs = [
                        (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
                        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)
                    ]
                    distances = []
                    for label_idx, detect_idx in keypoint_pairs:
                        if label_idx < len(keypoints) and detect_idx < len(keypoints_pixel):
                            lx, ly, lv = keypoints[label_idx]
                            if lv > 0:  # Only consider labeled keypoints with visibility > 0
                                dx, dy = keypoints_pixel[detect_idx]
                                distance = np.sqrt((lx - dx) ** 2 + (ly - dy) ** 2)
                                distances.append(distance)
                    
                    if distances:
                        mean_distance = np.mean(distances)
                        print(f"Mean Distance: {mean_distance}")
                    
                    cv2.imshow('Image with Keypoints', img)
                    cv2.waitKey(0)  # Press any key to close the image window
                    cv2.destroyAllWindows()
                else:
                    print(f"Unable to load image: {img_path}")
            else:
                print("No annotation found for this image.")
        else:
            print(f"Image: {image['file_name']} is not labeled.")

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
json_files = [f'injuryski_{i}.json' for i in range(1001, 1007)]

# Process each JSON file
for json_file in json_files:
    file_path = os.path.join(annotations_directory, json_file)
    if os.path.exists(file_path):
        process_json_file(file_path, base_directory, model)
    else:
        print(f"File {json_file} does not exist in the annotations directory.")
