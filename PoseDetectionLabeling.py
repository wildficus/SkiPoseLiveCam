import json
import os
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# Load an official model
model = YOLO('yolov8l-pose.pt')  # Load an official model

# Base directory containing the annotations and images
base_directory = 'C:/Users/maria/Desktop/Master/Dizertatie/injuryski_dataset/injuryski_dataset'
annotations_directory = os.path.join(base_directory, 'annotations')

# List of JSON files to process in the annotations directory
json_files = [f'injuryski_{i}.json' for i in range(1000, 1006)]
label_file = "labels.txt"

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

# Function to compute the angle formed by three points A, B, and C at point B
def compute_angle(pointA, pointB, pointC):
    if pointA is not None and pointB is not None and pointC is not None:
        A = np.array(pointA)
        B = np.array(pointB)
        C = np.array(pointC)
        BA = A - B
        BC = C - B
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)
        angle = np.arccos(dot_product / (magnitude_BA * magnitude_BC))
        angle_degrees = np.degrees(angle)
        return angle_degrees
    else:
        return -1

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    if point1 is not None and point2 is not None:
        p1 = np.array(point1)
        p2 = np.array(point2)
        distance = np.linalg.norm(p1 - p2)
        return distance
    else:
        return -1

# Function to plot keypoints and bones on the image
def plot_keypoints(image, keypoints_pixel):
    pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for i, (x, y) in enumerate(keypoints_pixel):
        ax.scatter(x, y, s=10, color='green')
        ax.text(x, y, str(i), color='blue', fontsize=4)

    for (i, j) in pairs:
        if (keypoints_pixel[i][0] != 0 or keypoints_pixel[i][1] != 0) and (keypoints_pixel[j][0] != 0 or keypoints_pixel[j][1] != 0):
            x1, y1 = int(keypoints_pixel[i][0]), int(keypoints_pixel[i][1])
            x2, y2 = int(keypoints_pixel[j][0]), int(keypoints_pixel[j][1])
            ax.plot([x1, x2], [y1, y2], color='red')

    plt.axis('off')
    return fig, ax

# Function to calculate angles and print the information
def calculate_angles(p_pixel, p_norm):
    if p_pixel is not None and p_pixel.shape[0] > 16 and p_norm is not None and p_norm.shape[0] > 16:
        nose, left_eye, left_ear, right_eye, right_ear = p_pixel[0], p_pixel[1], p_pixel[3], p_pixel[2], p_pixel[4]
        left_shld, left_elb, left_wrst, right_shld, right_elb, right_wrst = p_pixel[5], p_pixel[7], p_pixel[9], p_pixel[6], p_pixel[8], p_pixel[10]
        left_hip, left_knee, left_ankl, right_hip, right_knee, right_ankl = p_pixel[11], p_pixel[13], p_pixel[15], p_pixel[12], p_pixel[14], p_pixel[16]

        nose_n, left_eye_n, left_ear_n, right_eye_n, right_ear_n = p_norm[0], p_norm[1], p_norm[3], p_norm[2], p_norm[4]
        left_shld_n, left_elb_n, left_wrst_n, right_shld_n, right_elb_n, right_wrst_n = p_norm[5], p_norm[7], p_norm[9], p_norm[6], p_norm[8], p_norm[10]
        left_hip_n, left_knee_n, left_ankl_n, right_hip_n, right_knee_n, right_ankl_n = p_norm[11], p_norm[13], p_norm[15], p_norm[12], p_norm[14], p_norm[16]

        linie = "------------------------------------------------------"

        print(linie)
        
        # Flexarea membrelor pentru a mentine echilibrul si controlul
        angle1 = compute_angle(left_shld, left_elb, left_wrst)
        info_mana_stanga = f"Unghi mana stanga: {angle1} grade"
        print(info_mana_stanga)
                
        angle2 = compute_angle(right_shld, right_elb, right_wrst)
        info_mana_dreapta = f"Unghi mana dreapta: {angle2} grade"
        print(info_mana_dreapta)

        angle3 = compute_angle(left_hip, left_knee, left_ankl)
        info_picior_stang = f"Unghi picior stang: {angle3} grade"
        print(info_picior_stang)
        
        angle4 = compute_angle(right_hip, right_knee, right_ankl)
        info_picior_drept = f"Unghi picior drept: {angle4} grade"
        print(info_picior_drept)
        
        if np.isnan(angle1) or np.isnan(angle2) or np.isnan(angle3) or np.isnan(angle4):
            print("Membrele nu sunt suficient de vizibile.")
        else:
            if angle1 > 155 or angle2 > 155 or angle3 > 170 or angle4 > 170 or angle1 < 60 or angle2 < 60 or angle3 < 125 or angle4 < 125:
                print("Flexare membrelor este incorecta, risc de dezechilibru!")
            else:
                print("Postura este suficient de flexata.")
        
        print(linie)
        
        # Inclinarea laterala a corpului verifica postura suficient de dreapta
        angle5 = compute_angle(nose, left_hip, left_ankl)
        info_aplecare_stanga = f"Unghi aplecare 1 (stanga): {angle5} grade"
        print(info_aplecare_stanga)
        
        angle6 = compute_angle(nose, right_hip, right_ankl)
        info_aplecare_dreapta = f"Unghi aplecare 2 (dreapta): {angle6} grade"
        print(info_aplecare_dreapta)
        
        if np.isnan(angle5) or np.isnan(angle6) or np.array_equal(nose, [0, 0]) or np.array_equal(left_hip, [0, 0]) or np.array_equal(left_ankl, [0, 0]) or np.array_equal(right_hip, [0, 0]) or np.array_equal(right_ankl, [0, 0]):
            print("Inclinarea nu este suficient de clara.")
        else:
            if angle5 < 145 or angle6 < 145:
                print("Inclinare periculoasa!")
            else:
                print("Echilibru corect.")
        
        print(linie)
        
        # Distanta picioarelor raportata la distanta dintre coapse pentru a verifica pozitia paralela a schiurilor
        text_glezna_stanga = f"Glezna stanga norm: {left_ankl_n}"
        print(text_glezna_stanga)
        text_glezna_dreapta = f"Glezna dreapta norm: {right_ankl_n}"
        print(text_glezna_dreapta)
        
        dist_glezne = calculate_distance(left_ankl_n, right_ankl_n)
        dist_solduri = calculate_distance(left_hip_n, right_hip_n)
        text_dist_glezne = f"Distanta dintre cele doua glezne este: {dist_glezne}"
        text_dist_solduri = f"Distanta dintre cele doua solduri este: {dist_solduri}"
        print(text_dist_glezne)
        print(text_dist_solduri)
        
        if np.isnan(dist_glezne) or (dist_glezne == 0):
            print("Gleznele nu sunt suficient de vizibile!")
        else:
            if dist_glezne > (1.6 * dist_solduri):
                print("Schiurile sunt prea departate!")
            else:
                print("Schiurile sunt suficient de paralele.")
        
        print(linie)
        
        # Detectarea dezechilbrului/caderilor
        if left_wrst_n[1] == 0 or left_shld_n[1] == 0 or right_wrst_n[1] == 0 or right_shld_n[1] == 0: 
            print("Partea superioara a corpului nu este complet vizibila.")
        else:
            if left_wrst_n[1] <= left_shld_n[1] or right_wrst_n[1] <= right_shld_n[1]:
                print("Mainile sunt prea sus! Dezechilibrare detectata.")
            else:
                print("Elevatie corespunzatoare a mainilor.")
        
        print(linie)
        
        if left_knee_n[1] == 0 or right_knee_n[1] == 0 or left_hip_n[1] == 0 or right_hip_n[1] == 0:
            print("Partea inferioara a corpului nu este complet vizibila.")
        else:
            if min(left_knee_n[1], right_knee_n[1]) <= max(left_hip_n[1], right_hip_n[1]):
                print("Posteriorul este prea jos! Dezechilibrare detectata.")
            else:
                print("Elevatie corespunzatoare a posteriorului.")
        
        print(linie)

        # Check if the ankles are higher than any other keypoint except each other and the hands
        if left_ankl_n[1] != 0 or right_ankl_n[1] != 0:
            non_ankle_hand_points = [nose_n, left_eye_n, right_eye_n, left_ear_n, right_ear_n,
                                     left_shld_n, right_shld_n, left_elb_n, right_elb_n,
                                     left_hip_n, right_hip_n, left_knee_n, right_knee_n]
            
            max_non_ankle_hand_y = max(y for x, y in non_ankle_hand_points if y != 0)         
            if max_non_ankle_hand_y > left_ankl_n[1] or max_non_ankle_hand_y > right_ankl_n[1]:
                print(">>>>>>>>Cadere detectata!<<<<<<<<<<<<")
            else:
                print("Nicio cadere detectata")
    else:
        print("Array ended unexpectedly.")

# Function to process each JSON file
def process_json_file(file_path, base_directory, label_file):
    with open(file_path, 'r') as file:
        data = json.load(file)

    labels = {}
    if os.path.exists(label_file):
        with open(label_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(',')
                labels[parts[0]] = parts[1:]

    for image in data['images']:
        if image['is_labeled']:
            image_id = image['frame_id']
            annotation = next((ann for ann in data['annotations'] if ann['image_id'] == image_id), None)
            if annotation:
                keypoints = annotation.get('keypoints', [])
                img_path = os.path.join(base_directory, image['file_name'])
                img = cv2.imread(img_path)
                if img is not None:
                    results = model(img)
                    keypoints_pixel, keypoints_norm = extract_results(results)
                    if len(keypoints_norm) == 0:
                        print(f"No keypoints detected for image {image_id}")
                        continue

                    calculate_angles(keypoints_pixel, keypoints_norm)
                    fig, ax = plot_keypoints(img, keypoints_pixel)

                    ax_checkbox = plt.axes([0.75, 0.4, 0.2, 0.2])
                    check = CheckButtons(ax_checkbox, ['Loss of Balance', 'Fall'], [False, False])

                    def submit(event):
                        loss_of_balance = check.get_status()[0]
                        fall = check.get_status()[1]
                        labels[img_path] = [str(int(loss_of_balance)), str(int(fall))]
                        with open(label_file, 'a') as file:
                            file.write(f"{img_path},{int(loss_of_balance)},{int(fall)}\n")
                        plt.close()

                    fig.canvas.mpl_connect('key_press_event', lambda event: submit(event))
                    plt.show()

# Process each JSON file
for json_file in json_files:
    file_path = os.path.join(annotations_directory, json_file)
    if os.path.exists(file_path):
        process_json_file(file_path, base_directory, label_file)
    else:
        print(f"File {json_file} does not exist in the annotations directory.")
