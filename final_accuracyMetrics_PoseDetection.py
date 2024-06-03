import json
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score

#---------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- Functions used -------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#

def compute_angle(pointA, pointB, pointC):
    """
    Compute the angle formed by three points A, B, and C at point B.

    Parameters:
    pointA, pointB, pointC: The coordinates of the points (x, y), as tuples or lists.

    Returns:
    The angle in degrees.
    """
    if pointA is not None and pointB is not None and pointC is not None:
        A = np.array(pointA)
        B = np.array(pointB)
        C = np.array(pointC)

        # Create vectors
        BA = A - B
        BC = C - B

        # Compute the dot product and magnitudes of the vectors
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)

        if magnitude_BA == 0 or magnitude_BC == 0:
            return None

        # Calculate the angle using the dot product formula
        cos_angle = dot_product / (magnitude_BA * magnitude_BC)

        # Ensure the value is within the valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)

        # Convert the angle to degrees
        angle_degrees = np.degrees(angle)

        return angle_degrees
    else:
        return None

def extract_results(kpts):
    # Process results list
    for result in kpts:
        keypoints = result.keypoints  # Keypoints object for pose outputs    
    # Convert keypoints to a NumPy array
    # Since we have only one person, access the keypoints for that person
    keypoints_pixel = keypoints.data.numpy()[0]
    keypoints_pixel = np.delete(keypoints_pixel, 2, 1)  
    keypoints_norm  = keypoints.xyn.numpy()[0] 
    return [keypoints_pixel, keypoints_norm]

def calculate_distance(point1, point2):
    """
    Euclidian distance between two points

    Parameters:
    point1, point2

    Returns:
    Euclidian distance
    """
    if point1 is not None and point2 is not None:
        p1 = np.array(point1)
        p2 = np.array(point2)
        distance = np.linalg.norm(p1 - p2)
        return distance
    else:
        return -1

def calculate_angles(p_pixel, p_norm):
    if p_pixel is not None and p_pixel.shape[0] > 16 and p_norm is not None and p_norm.shape[0] > 16:
        nose        = p_pixel[0]
        left_eye    = p_pixel[1]
        left_ear    = p_pixel[3]
        right_eye   = p_pixel[2]
        right_ear   = p_pixel[4]
        left_shld   = p_pixel[5]
        left_elb    = p_pixel[7]
        left_wrst   = p_pixel[9]
        right_shld  = p_pixel[6]
        right_elb   = p_pixel[8]
        right_wrst  = p_pixel[10]
        left_hip    = p_pixel[11]
        left_knee   = p_pixel[13]
        left_ankl   = p_pixel[15]
        right_hip   = p_pixel[12]
        right_knee  = p_pixel[14]
        right_ankl  = p_pixel[16]

        nose_n        = p_norm[0]
        left_eye_n    = p_norm[1]
        left_ear_n    = p_norm[3]
        right_eye_n   = p_norm[2]
        right_ear_n   = p_norm[4]
        left_shld_n   = p_norm[5]
        left_elb_n    = p_norm[7]
        left_wrst_n   = p_norm[9]
        right_shld_n  = p_norm[6]
        right_elb_n   = p_norm[8]
        right_wrst_n  = p_norm[10]
        left_hip_n    = p_norm[11]
        left_knee_n   = p_norm[13]
        left_ankl_n   = p_norm[15]
        right_hip_n   = p_norm[12]
        right_knee_n  = p_norm[14]
        right_ankl_n  = p_norm[16]

        balance_flexion = 0
        balance_inclination = 0
        balance_legs_parallel = 0
        balance_behind = 0
        balance_hands = 0
        fall_ankle = 0
        fall_nose = 0

        angle1 = compute_angle(left_shld, left_elb, left_wrst)
        angle2 = compute_angle(right_shld, right_elb, right_wrst)
        angle3 = compute_angle(left_hip, left_knee, left_ankl)
        angle4 = compute_angle(right_hip, right_knee, right_ankl)

        if (angle1 is None or angle2 is None or angle3 is None or angle4 is None or
                np.isnan(angle1) or np.isnan(angle2) or np.isnan(angle3) or np.isnan(angle4)):
            balance_flexion = 0
        else:
            if angle1 > 155 or angle2 > 155 or angle3 > 170 or angle4 > 170 or angle1 < 60 or angle2 < 60 or angle3 < 125 or angle4 < 125:
                balance_flexion = 1

        angle5 = compute_angle(nose, left_hip, left_ankl)
        angle6 = compute_angle(nose, right_hip, right_ankl)

        if (angle5 is None or angle6 is None or np.isnan(angle5) or np.isnan(angle6) or
                np.array_equal(nose, [0, 0]) or np.array_equal(left_hip, [0, 0]) or np.array_equal(left_ankl, [0, 0]) or
                np.array_equal(right_hip, [0, 0]) or np.array_equal(right_ankl, [0, 0])):
            balance_inclination = 0
        else:
            if angle5 < 145 or angle6 < 145:
                balance_inclination = 1

        dist_glezne = calculate_distance(left_ankl_n, right_ankl_n)
        dist_solduri = calculate_distance(left_hip_n, right_hip_n)

        if (np.isnan(dist_glezne) or dist_glezne == 0 or np.isnan(dist_solduri) or dist_solduri == 0 or 
            left_hip_n[1] == 0 or right_hip_n[1] == 0):
            balance_legs_parallel = 0
        else:
            if dist_glezne > (1.6 * dist_solduri):
                balance_legs_parallel = 1

        if left_knee_n[1] == 0 or right_knee_n[1] == 0 or left_hip_n[1] == 0 or right_hip_n[1] == 0:
            balance_behind = 0
        else:
            if min(left_knee_n[1], right_knee_n[1]) <= max(left_hip_n[1], right_hip_n[1]):
                balance_behind = 1

        if left_wrst_n[1] == 0 or left_shld_n[1] == 0 or right_wrst_n[1] == 0 or right_shld_n[1] == 0:
            balance_hands = 0
        else:
            if left_wrst_n[1] <= left_shld_n[1] or right_wrst_n[1] <= right_shld_n[1]:
                balance_hands = 1

        non_ankle_hand_points = [nose_n, left_eye_n, right_eye_n, left_ear_n, right_ear_n,
                                 left_shld_n, right_shld_n, left_elb_n, right_elb_n,
                                 left_hip_n, right_hip_n, left_knee_n, right_knee_n]

        valid_y_values = [y for x, y in non_ankle_hand_points if y != 0]

        if valid_y_values:
            max_non_ankle_hand_y = max(valid_y_values)
        else:
            max_non_ankle_hand_y = 0

        if left_ankl_n[1] != 0 or right_ankl_n[1] != 0:
            if max_non_ankle_hand_y > left_ankl_n[1] or max_non_ankle_hand_y > right_ankl_n[1]:
                fall_ankle = 1

        if nose_n[1] != 0:
            other_points = [left_hip_n, right_hip_n, left_knee_n, right_knee_n, left_ankl_n, right_ankl_n, left_shld_n, right_shld_n]
            if any(nose_n[1] > pt[1] for pt in other_points if pt[1] != 0):
                fall_nose = 1

        # Determine loss_of_balance and fall_detected flags
        balance_flags = [balance_flexion, balance_inclination, balance_legs_parallel, balance_behind, balance_hands]
        fall_flags = [fall_ankle, fall_nose]

        loss_of_balance = sum(balance_flags) >= 3
        fall_detected = any(fall_flags)

        return {
            'balance_flexion': balance_flexion,
            'balance_inclination': balance_inclination,
            'balance_legs_parallel': balance_legs_parallel,
            'balance_behind': balance_behind,
            'balance_hands': balance_hands,
            'fall_ankle': fall_ankle,
            'fall_nose': fall_nose,
            'loss_of_balance': loss_of_balance,
            'fall_detected': fall_detected
        }

    else:
        return {'error': "Array ended unexpectedly."}

#---------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Main body ----------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#

# Load JSON file with image paths
json_file_path = 'C:/Users/maria/Desktop/Master/Dizertatie/injuryski_dataset/injuryski_dataset/poseLabels.json'  # Replace with the actual path to your JSON file

with open(json_file_path, 'r') as file:
    data = json.load(file)

# Load YOLO model
model = YOLO('yolov8l-pose.pt')

# Initialize lists for true and predicted labels
true_labels_balance = []
pred_labels_balance = []
true_labels_fall = []
pred_labels_fall = []

# Iterate over each image in the JSON file
for entry in data:
    image_path = entry['image_path']
    image = cv.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        continue
    
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(image_rgb)
    
    # Extract keypoints
    [keypoints_pixel, keypoints_norm] = extract_results(results)
    
    # Calculate angles and other metrics
    metrics = calculate_angles(keypoints_pixel, keypoints_norm)
    
    # Compare the labeled flags from JSON with the computed results
    labeled_loss_of_balance = entry.get('loss_of_balance', None)
    labeled_fall_detected = entry.get('fall_detected', None)
    
    # Collect true and predicted labels for loss of balance
    if labeled_loss_of_balance is not None and 'loss_of_balance' in metrics:
        true_labels_balance.append(labeled_loss_of_balance)
        pred_labels_balance.append(metrics['loss_of_balance'])
    
    # Collect true and predicted labels for fall detection
    if labeled_fall_detected is not None and 'fall_detected' in metrics:
        true_labels_fall.append(labeled_fall_detected)
        pred_labels_fall.append(metrics['fall_detected'])
    
    print(f"Metrics for {image_path}: {metrics}")
    print(f"Labeled loss of balance: {labeled_loss_of_balance}, Computed loss of balance: {metrics.get('loss_of_balance')}")
    print(f"Labeled fall detected: {labeled_fall_detected}, Computed fall detected: {metrics.get('fall_detected')}")

# Calculate accuracy metrics for loss of balance
if true_labels_balance and pred_labels_balance:
    conf_matrix_balance = confusion_matrix(true_labels_balance, pred_labels_balance)
    precision_balance = precision_score(true_labels_balance, pred_labels_balance)
    recall_balance = recall_score(true_labels_balance, pred_labels_balance)
    print("\nLoss of Balance:")
    print("Confusion Matrix:")
    print(conf_matrix_balance)
    print(f"Precision: {precision_balance}")
    print(f"Recall: {recall_balance}")

# Calculate accuracy metrics for fall detection
if true_labels_fall and pred_labels_fall:
    conf_matrix_fall = confusion_matrix(true_labels_fall, pred_labels_fall)
    precision_fall = precision_score(true_labels_fall, pred_labels_fall)
    recall_fall = recall_score(true_labels_fall, pred_labels_fall)
    print("\nFall Detection:")
    print("Confusion Matrix:")
    print(conf_matrix_fall)
    print(f"Precision: {precision_fall}")
    print(f"Recall: {recall_fall}")
