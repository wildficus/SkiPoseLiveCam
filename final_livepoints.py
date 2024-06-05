from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2 as cv

#---------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------- Global vars ----------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#

nose        = None
left_eye    = None
left_ear    = None
right_eye   = None
right_ear   = None
left_shld   = None
left_elb    = None
left_wrst   = None
right_shld  = None
right_elb   = None
right_wrst  = None
left_hip    = None
left_knee   = None
left_ankl   = None
right_hip   = None
right_knee  = None
right_ankl  = None

nose_n        = None
left_eye_n    = None
left_ear_n    = None
right_eye_n   = None
right_ear_n   = None
left_shld_n   = None
left_elb_n    = None
left_wrst_n   = None
right_shld_n  = None
right_elb_n   = None
right_wrst_n  = None
left_hip_n    = None
left_knee_n   = None
left_ankl_n   = None
right_hip_n   = None
right_knee_n  = None
right_ankl_n  = None

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
    keypoints_pixel = np.delete(keypoints_pixel,2,1)  
    keypoints_norm  = keypoints.xyn.numpy()[0] 
    return [keypoints_pixel, keypoints_norm]

def plot_photo(img, results):
    [keypoints_pixel,keypoints_norm] = extract_results(results)
    
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    # Plot and annotate each keypoint
    for i, (x, y, conf) in enumerate(keypoints_pixel):
        ax.scatter(x, y, s=50, color='red')  # Plot keypoint
        ax.text(x, y, str(i), color='blue', fontsize=12)  # Annotate keypoint index

    plt.axis('off')
    plt.show()

def plot_frame(frame_to_plot, points_struct):
    if points_struct is not None and points_struct.shape[0] > 16:
        for point in points_struct:
            if point[0] != 0 or point[1] != 0:  # Check if point is detected
                x, y = int(point[0]), int(point[1])
                cv.circle(frame_to_plot, (x, y), 2, (0, 255, 0), -1)  # Draw circle

        pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
        for (i, j) in pairs:
            if (points_struct[i][0] != 0 or points_struct[i][1] != 0) and (points_struct[j][0] != 0 or points_struct[j][1] != 0):
                x1, y1 = int(points_struct[i][0]), int(points_struct[i][1])
                x2, y2 = int(points_struct[j][0]), int(points_struct[j][1])
                cv.line(frame_to_plot, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
        
        linie = "-------------------------------------------------------------------------------"
        
        cv.line(blank_image, (480, 60), (480, 450), (0, 0, 0), 1)
        
        print(linie)
        cv.line(blank_image, (10, 60), (950, 60), (0, 0, 0), 2)
        
        angle1 = compute_angle(left_shld, left_elb, left_wrst)
        if angle1 is not None and not np.isnan(angle1):
            info_mana_stanga = f"Unghi mana stanga: {round(angle1)} grade"
        else:
            info_mana_stanga = "Unghi mana stanga: N/A"
        print(info_mana_stanga)
        cv.putText(blank_image, info_mana_stanga, (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        angle2 = compute_angle(right_shld, right_elb, right_wrst)
        if angle2 is not None and not np.isnan(angle2):
            info_mana_dreapta = f"Unghi mana dreapta: {round(angle2)} grade"
        else:
            info_mana_dreapta = "Unghi mana dreapta: N/A"
        print(info_mana_dreapta)
        cv.putText(blank_image, info_mana_dreapta, (10, 95), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        angle3 = compute_angle(left_hip, left_knee, left_ankl)
        if angle3 is not None and not np.isnan(angle3):
            info_picior_stang = f"Unghi picior stang: {round(angle3)} grade"
        else:
            info_picior_stang = "Unghi picior stang: N/A"
        print(info_picior_stang)
        cv.putText(blank_image, info_picior_stang, (10, 115), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        angle4 = compute_angle(right_hip, right_knee, right_ankl)
        if angle4 is not None and not np.isnan(angle4):
            info_picior_drept = f"Unghi picior drept: {round(angle4)} grade"
        else:
            info_picior_drept = "Unghi picior drept: N/A"
        print(info_picior_drept)
        cv.putText(blank_image, info_picior_drept, (10, 135), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        if (angle1 is None or angle2 is None or angle3 is None or angle4 is None or
                np.isnan(angle1) or np.isnan(angle2) or np.isnan(angle3) or np.isnan(angle4)):
            cv.putText(blank_image, "Membrele nu sunt suficient de vizibile.", (485, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            print("Membrele nu sunt suficient de vizibile.")
        else:
            if angle1 > 155 or angle2 > 155 or angle3 > 170 or angle4 > 170 or angle1 < 60 or angle2 < 60 or angle3 < 125 or angle4 < 125:
                cv.putText(blank_image, "Flexare membrelor este incorecta, risc de dezechilibru!", (485, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                print("Flexare membrelor este incorecta, risc de dezechilibru!")
            else:
                cv.putText(blank_image, "Postura este suficient de flexata.", (485, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                print("Postura este suficient de flexata.")

        print(linie)
        cv.line(blank_image, (10, 155), (950, 155), (0, 0, 0), 1)

        angle5 = compute_angle(nose, left_hip, left_ankl)
        if angle5 is not None and not np.isnan(angle5):
            info_aplecare_stanga = f"Unghi aplecare 1 (stanga): {round(angle5)} grade"
        else:
            info_aplecare_stanga = "Unghi aplecare 1 (stanga): N/A"
        print(info_aplecare_stanga)
        cv.putText(blank_image, info_aplecare_stanga, (10, 170), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        angle6 = compute_angle(nose, right_hip, right_ankl)
        if angle6 is not None and not np.isnan(angle6):
            info_aplecare_dreapta = f"Unghi aplecare 2 (dreapta): {round(angle6)} grade"
        else:
            info_aplecare_dreapta = "Unghi aplecare 2 (dreapta): N/A"
        print(info_aplecare_dreapta)
        cv.putText(blank_image, info_aplecare_dreapta, (10, 190), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        if (angle5 is None or angle6 is None or np.isnan(angle5) or np.isnan(angle6) or
                np.array_equal(nose, [0, 0]) or np.array_equal(left_hip, [0, 0]) or np.array_equal(left_ankl, [0, 0]) or
                np.array_equal(right_hip, [0, 0]) or np.array_equal(right_ankl, [0, 0])):
            print("Inclinarea nu este suficient de clara.")
            cv.putText(blank_image, "Inclinarea nu este suficient de clara.", (485, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        else:
            if angle5 < 145 or angle6 < 145:
                print("Inclinare periculoasa!")
                cv.putText(blank_image, "Inclinare periculoasa!", (485, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            else:
                print("Echilibru corect.")
                cv.putText(blank_image, "Echilibru corect", (485, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        print(linie)
        cv.line(blank_image, (10, 205), (950, 205), (0, 0, 0), 1)

        text_glezna_stanga = f"Glezna stanga norm: {left_ankl_n[1]:.2f}"
        print(text_glezna_stanga)
        cv.putText(blank_image, text_glezna_stanga, (10, 220), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        text_glezna_dreapta = f"Glezna dreapta norm: {right_ankl_n[1]:.2f}"
        print(text_glezna_dreapta)
        cv.putText(blank_image, text_glezna_dreapta, (10, 235), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        dist_glezne = calculate_distance(left_ankl_n, right_ankl_n)
        dist_solduri = calculate_distance(left_hip_n, right_hip_n)
        text_dist_glezne = f"Distanta dintre cele doua glezne este: {dist_glezne:.2f}"
        text_dist_solduri = f"Distanta dintre cele doua solduri este: {dist_solduri:.2f}"
        print(text_dist_glezne)
        cv.putText(blank_image, text_dist_glezne, (10, 250), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        print(text_dist_solduri)
        cv.putText(blank_image, text_dist_solduri, (10, 265), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        if np.isnan(dist_glezne) or (dist_glezne == 0):
            print("Gleznele nu sunt suficient de vizibile!")
            cv.putText(blank_image, "Gleznele nu sunt suficient de vizibile!", (485, 240), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        else:
            if dist_glezne > (1.6 * dist_solduri):
                print("Schiurile sunt prea departate!")
                cv.putText(blank_image, "Schiurile sunt prea departate!", (485, 240), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            else:
                print("Schiurile sunt suficient de paralele.")
                cv.putText(blank_image, "Schiurile sunt suficient de paralele.", (485, 240), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        print(linie)
        cv.line(blank_image, (10, 290), (950, 290), (0, 0, 0), 1)

        if left_wrst_n[1] == 0 or left_shld_n[1] == 0 or right_wrst_n[1] == 0 or right_shld_n[1] == 0:
            print("Partea superioara a corpului nu este complet vizibila.")
            cv.putText(blank_image, "Partea superioara a corpului nu este complet vizibila.", (485, 305), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        else:
            if left_wrst_n[1] <= left_shld_n[1] or right_wrst_n[1] <= right_shld_n[1]:
                print("Mainile sunt prea sus! Dezechilibrare detectata.")
                cv.putText(blank_image, "Mainile sunt prea sus! Dezechilibrare detectata.", (485, 305), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            else:
                print("Elevatie corespunzatoare a mainilor.")
                cv.putText(blank_image, "Elevatie corespunzatoare a mainilor.", (485, 305), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        print(linie)
        cv.line(blank_image, (10, 330), (950, 330), (0, 0, 0), 1)

        if left_knee_n[1] == 0 or right_knee_n[1] == 0 or left_hip_n[1] == 0 or right_hip_n[1] == 0:
            print("Partea inferioara a corpului nu este complet vizibila.")
            cv.putText(blank_image, "Partea inferioara a corpului nu este complet vizibila.", (485, 345), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        else:
            if min(left_knee_n[1], right_knee_n[1]) <= max(left_hip_n[1], right_hip_n[1]):
                print("Bazinul este prea jos! Dezechilibrare detectata.")
                cv.putText(blank_image, "Bazinul este prea jos! Dezechilibrare detectata.", (485, 345), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            else:
                print("Elevatie corespunzatoare a bazinului.")
                cv.putText(blank_image, "Elevatie corespunzatoare a bazinului.", (485, 345), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        print(linie)
        cv.line(blank_image, (10, 370), (950, 370), (0, 0, 0), 1)

        non_ankle_hand_points = [nose_n, left_eye_n, right_eye_n, left_ear_n, right_ear_n,
                                 left_shld_n, right_shld_n, left_elb_n, right_elb_n,
                                 left_hip_n, right_hip_n, left_knee_n, right_knee_n]

        valid_y_values = [y for x, y in non_ankle_hand_points if y != 0]

        if valid_y_values:
            max_non_ankle_hand_y = max(valid_y_values)
        else:
            max_non_ankle_hand_y = 0  # or some default value

#        cv.putText(blank_image, f"left_ankl_n: {left_ankl_n[1]:.2f} ", (10, 410), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
#        cv.putText(blank_image, f"right_ankl_n: {right_ankl_n[1]:.2f} ", (10, 425), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
#        cv.putText(blank_image, f"max_non_ankle_hand_y: {max_non_ankle_hand_y:.2f} ", (10, 440), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            
        if left_ankl_n[1] != 0 or right_ankl_n[1] != 0:
            if max_non_ankle_hand_y > left_ankl_n[1] or max_non_ankle_hand_y > right_ankl_n[1]:
                print("Cadere detectata!(picioare)")
                cv.putText(blank_image, "Cadere detectata!(picioare)", (485, 395), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            else:
                print("Pozitia picioareler corecta")
                cv.putText(blank_image, "Pozitia picioareler corecta", (485, 395), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        if nose_n[1] != 0:
            other_points = [left_hip_n, right_hip_n, left_knee_n, right_knee_n, left_ankl_n, right_ankl_n, left_shld_n, right_shld_n]
            if any(nose_n[1] > pt[1] for pt in other_points if pt[1] != 0):
                print("Cadere detectata!(cap)")
                cv.putText(blank_image, "Cadere detectata!(cap)", (485, 410), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            else:
                print("Pozitia capului corecta")
                cv.putText(blank_image, "Pozitia capului corecta", (485, 410), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        
        cv.line(blank_image, (10, 450), (950, 450), (0, 0, 0), 1)
        
    else:
        print(f"Array ended unexpectedly.")

#---------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Main body ----------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

model = YOLO('yolov8l-pose.pt')

while True:
    ret, image = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    results = model(image)  

    [keypoints_pixel, keypoints_norm] = extract_results(results)
    
    plot_frame(image,keypoints_pixel)
    
  # Create a new blank image with extended width
    extended_width = 1.5 * image.shape[1]  # Extend the width by a factor of 2
    blank_image = np.full((image.shape[0], round(extended_width), image.shape[2]), 255, dtype=np.uint8)
    
    text_info = "Real-time Keypoint Information:"
    cv.putText(blank_image, text_info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
    cv.putText(blank_image, "(Press 'q' to quit)", (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv.LINE_AA)
    
    calculate_angles(keypoints_pixel,keypoints_norm)
    
    combined_image = np.concatenate((image, blank_image), axis=1)

    window_title = 'Ski Live Pose Cam'
    cv.imshow(window_title, combined_image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
