import numpy as np
import h5py
import imageio
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
import cv2 as cv

# Define keypoints of interest
my_keypoints = ['left_shld', 'left_elb', 'left_wrst', 'right_shld', 'right_elb', 'right_wrst', 'left_hip', 'left_knee', 'left_ankl', 'right_hip', 'right_knee', 'right_ankl']
joint_names_h36m = ['hip', 'right_up_leg', 'right_leg', 'right_foot', 'left_up_leg', 'left_leg', 'left_foot', 'spine1', 'neck', 'head', 'head-top', 'left-arm', 'left_forearm', 'left_hand', 'right_arm', 'right_forearm', 'right_hand']
bones_h36m = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]

# Mapping from SkiPosePTZ keypoints to your keypoints
ski_pose_to_my_pose = {
    'left_shld': 'left-arm',
    'left_elb': 'left_forearm',
    'left_wrst': 'left_hand',
    'right_shld': 'right_arm',
    'right_elb': 'right_forearm',
    'right_wrst': 'right_hand',
    'left_hip': 'left_up_leg',
    'left_knee': 'left_leg',
    'left_ankl': 'left_foot',
    'right_hip': 'right_up_leg',
    'right_knee': 'right_leg',
    'right_ankl': 'right_foot'
}

# Function to extract results from model predictions
def extract_results(kpts):
    for result in kpts:
        keypoints = result.keypoints  # Keypoints object for pose outputs    
    keypoints_pixel = keypoints.data.numpy()[0]
    keypoints_pixel = np.delete(keypoints_pixel, 2, 1)  
    keypoints_norm = keypoints.xyn.numpy()[0] 
    return [keypoints_pixel, keypoints_norm]

# Function to extract keypoints from h5 file
def extract_keypoints(h5_file, index):
    pose_2D = h5_file['2D'][index].reshape([-1, 2])  # in range 0..1
    pose_2D_px = 256 * pose_2D  # in pixels, range 0..255
    keypoints = {my_kpt: pose_2D_px[joint_names_h36m.index(ski_kpt)] for my_kpt, ski_kpt in ski_pose_to_my_pose.items()}
    return keypoints, pose_2D_px

# Function to calculate accuracy metrics
def calculate_accuracy(predicted_keypoints, true_keypoints):
    errors = []
    for kpt in predicted_keypoints.keys():
        pred = predicted_keypoints[kpt]
        true = true_keypoints[kpt]
        errors.append(np.linalg.norm(pred - true))
    return np.mean(errors), np.std(errors)

# Initialize the model
model = YOLO('yolov8l-pose.pt')

# Folder structure
base_folder = './test'
label_file_path = os.path.join(base_folder, 'labels.h5')

h5_label_file = h5py.File(label_file_path, 'r')
num_samples = len(h5_label_file['seq'])

# Iterate through each sample
for index in range(num_samples):
    seq = int(h5_label_file['seq'][index])
    cam = int(h5_label_file['cam'][index])
    frame = int(h5_label_file['frame'][index])

    # Extract true keypoints
    true_keypoints, pose_2D_px = extract_keypoints(h5_label_file, index)

    # Load image
    cam_folder_name = f'cam_{cam:02d}'
    image_name = os.path.join(base_folder, f'seq_{seq:03d}', cam_folder_name, f'image_{frame:06d}.png')
    if not os.path.isfile(image_name):
        continue

    img = imageio.imread(image_name)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    # Predict with the model on the frame
    results = model(img)
    keypoints_pixel, _ = extract_results(results)

    # Map predicted keypoints
    predicted_keypoints = {my_kpt: keypoints_pixel[list(ski_pose_to_my_pose.keys()).index(my_kpt)] for my_kpt in my_keypoints}

    # Calculate accuracy metrics
    mean_error, std_error = calculate_accuracy(predicted_keypoints, true_keypoints)
    print(f"Seq: {seq}, Cam: {cam}, Frame: {frame} - Mean Error: {mean_error}, Std Error: {std_error}")

    # Overlay keypoints on image
    for kpt in my_keypoints:
        pred = predicted_keypoints[kpt]
        true = true_keypoints[kpt]
        # Draw predicted keypoints in red with label
        cv.circle(img, (int(pred[0]), int(pred[1])), 2, (0, 0, 255), -1)
        cv.putText(img, kpt, (int(pred[0]), int(pred[1]) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        # Draw true keypoints in green with label
        cv.circle(img, (int(true[0]), int(true[1])), 2, (0, 255, 0), -1)
        cv.putText(img, kpt, (int(true[0]), int(true[1]) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Display image with overlayed keypoints
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_rgb)
    plt.title(f'Seq: {seq}, Cam: {cam}, Frame: {frame}')
    plt.axis('off')
    plt.show()
