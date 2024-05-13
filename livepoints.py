from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2 as cv
import numpy as np


# Example usage
# Normalized positions of the points (x, y)
# pointA = (0.5, 0.5)  # Replace with actual normalized positions
# pointB = (0.6, 0.6)
# pointC = (0.7, 0.5)
# Compute the angle
# angle = compute_angle(pointA, pointB, pointC)
# print(f"Angle at point B: {angle} degrees")
def compute_angle(pointA, pointB, pointC):
    """
    Compute the angle formed by three points A, B, and C at point B.

    Parameters:
    pointA, pointB, pointC: The coordinates of the points (x, y), as tuples or lists.

    Returns:
    The angle in degrees.
    """
    # Convert points to numpy arrays
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

    # Calculate the angle using the dot product formula
    angle = np.arccos(dot_product / (magnitude_BA * magnitude_BC))

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle)

    return angle_degrees


# Iterate through the results tensor and select the only person hopefully 
# (if the photo does not contain more than one person)
# Input the results 
def extract_results(kpts):
    # Process results list
    for result in kpts:
        keypoints = result.keypoints  # Keypoints object for pose outputs    
    # Convert keypoints to a NumPy array
    # Since we have only one person, access the keypoints for that person
    keypoints_pixel = keypoints.data.numpy()[0]   
    keypoints_norm  = keypoints.xyn.numpy()[0] 
    return [keypoints_pixel, keypoints_norm]
    
    

def plot_photo(img, results):
      
    # Load your image
    #loaded_img = Image.open(img)

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

def calculate_distance(point1, point2):
    """
    Euclidian distance between two points

    Parameters:
    point1, point2

    Returns:
    Euclidian distance
    """
    # Convert points into numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)

    # Calculate distance
    distance = np.linalg.norm(p1 - p2)

    return distance
    
def dump_keypoints(keypoints_pixel, keypoints_norm):
    
    if keypoints_pixel is not None and keypoints_pixel.shape[0] > 16 and keypoints_norm is not None and keypoints_norm.shape[0] > 16:
        nose        = keypoints_pixel[0]

        left_eye    = keypoints_pixel[1]
        left_ear    = keypoints_pixel[3]

        right_eye   = keypoints_pixel[2]
        right_ear   = keypoints_pixel[4]

        left_shld   = keypoints_pixel[5]
        left_elb    = keypoints_pixel[7]
        left_wrst   = keypoints_pixel[9]

        right_shld  = keypoints_pixel[6]
        right_elb   = keypoints_pixel[8]
        right_wrst  = keypoints_pixel[10]

        left_hip    = keypoints_pixel[11]
        left_knee   = keypoints_pixel[13]
        left_ankl   = keypoints_pixel[15]

        right_hip   = keypoints_pixel[12]
        right_knee  = keypoints_pixel[14]
        right_ankl  = keypoints_pixel[16]


        nose_n        = keypoints_norm[0]

        left_eye_n    = keypoints_norm[1]
        left_ear_n    = keypoints_norm[3]

        right_eye_n   = keypoints_norm[2]
        right_ear_n   = keypoints_norm[4]

        left_shld_n   = keypoints_norm[5]
        left_elb_n    = keypoints_norm[7]
        left_wrst_n   = keypoints_norm[9]

        right_shld_n  = keypoints_norm[6]
        right_elb_n   = keypoints_norm[8]
        right_wrst_n  = keypoints_norm[10]

        left_hip_n    = keypoints_norm[11]
        left_knee_n   = keypoints_norm[13]
        left_ankl_n   = keypoints_norm[15]

        right_hip_n   = keypoints_norm[12]
        right_knee_n  = keypoints_norm[14]
        right_ankl_n  = keypoints_norm[16]
        
        print(f"Nas pixel:                  {nose},|\nNas normat:                   {nose_n}")
        print(f"Ochi stang pixel:           {left_eye},|\nOchi stang normat:            {left_eye_n}")
        print(f"Ochi drept pixel:           {right_eye},|\nOchi drept normat:            {right_eye}")
        print(f"Ureche stanga pixel:        {left_ear},|\nUreche stanga normat:         {left_ear_n}")
        print(f"Ureche dreapta pixel:       {right_ear},|\nUreche dreapta normat:        {right_ear_n}")
        print(f"Umar stang pixel:           {left_shld},|\nUmar stang normat:            {left_shld_n}")
        print(f"Umar drept pixel:           {right_shld},|\nUmar drept normat:           {right_shld}")
        print(f"Cot stang pixel:            {left_elb},|\nCot stang normat:             {left_elb_n}")
        print(f"Cot drept pixel:            {right_elb},|\nCot drept normat:             {right_elb_n}")
        print(f"Incheietura stanga pixel:   {left_wrst},|\nIncheietura stanga normat:    {left_wrst_n}")
        print(f"Incheietura dreapta pixel:  {right_wrst},|\nIncheietura dreapta normat:   {right_wrst_n}")
        print(f"Sold stang pixel:           {left_hip},|\nSold stang normat:            {left_hip_n}")
        print(f"Sold drept pixel:           {right_hip},|\nSold drept normat:            {right_hip_n}")
        print(f"Genunchi stang pixel:       {left_knee},|\nGenunchi stang normat:        {left_knee_n}")
        print(f"Genunchi drept pixel:       {right_knee},|\nGenunchi drept normat:        {right_knee_n}")
        print(f"Glezna stanga pixel:        {left_ankl},|\nGlezna stanga normat:         {left_ankl_n}") 
        print(f"Glezna dreapta pixel:       {right_ankl},|\nGlezna dreapta normat:        {right_ankl_n}") 
    else:
        print(f"Array ended unexpectedly.")
    
    
    
    
    
    
    
    
    
    
    
    
    
#---------------------------------------------------------------------------------------------------------------------------#
#Load a photo
#img_name = '-000450.jpg'







cap = cv.VideoCapture(0)
# Load a model
model = YOLO('yolov8l-pose.pt')  # load an official model

if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    # Our operations on the frame come here
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Predict with the model
    results = model(image)  # predict on an image

    # Extract the keypoint positions (non normalised, pixel coordinates)    
    [keypoints_pixel,keypoints_norm] = extract_results(results)
    
    if keypoints_pixel is not None and keypoints_pixel.shape[0] > 16:
        for point in keypoints_pixel:
            if point[0] != 0 or point[1] != 0:  # Check if point is detected
                x, y = int(point[0]), int(point[1])
                cv.circle(image, (x, y), 2, (0, 255, 0), -1)  # Draw circle    

        # Define connections between keypoints
        pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
        for (i, j) in pairs:
            if (keypoints_pixel[i][0] != 0 or keypoints_pixel[i][1] != 0) and (keypoints_pixel[j][0] != 0 or keypoints_pixel[j][1] != 0):
                x1, y1 = int(keypoints_pixel[i][0]), int(keypoints_pixel[i][1])
                x2, y2 = int(keypoints_pixel[j][0]), int(keypoints_pixel[j][1])
                cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Create a blank image for the text (same size as the frame)
    blank_image = np.zeros_like(frame)
    
    # You can customize this text and the location as needed
    text_info = "Real-time Keypoint Information:"
    cv.putText(blank_image, text_info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(blank_image, "Press 'q' to quit", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)

    # Concatenate image horizontally
    combined_image = np.concatenate((image, blank_image), axis=1)
  
    # Display the resulting concatenated image
    cv.imshow('Video with Text Side by Side', combined_image)
    
    dump_keypoints(keypoints_pixel,keypoints_norm)
    
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()




# ret, frame = cap.read()
# # if frame is read correctly ret is True
# if not ret:
    # print("Can't receive frame (stream end?). Exiting ...")
# image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
# cap.release()



# # Predict with the model
# results = model(image)  # predict on an image

# # Plot image with keypoints
# plot_photo(image,results)

# # Extract the keypoint positions (non normalised, pixel coordinates)
# [keypoints_pixel,keypoints_norm] = extract_results(results)

# Transfer each keypoint to a natural language variable
nose        = keypoints_pixel[0]

left_eye    = keypoints_pixel[1]
left_ear    = keypoints_pixel[3]

right_eye   = keypoints_pixel[2]
right_ear   = keypoints_pixel[4]

left_shld   = keypoints_pixel[5]
left_elb    = keypoints_pixel[7]
left_wrst   = keypoints_pixel[9]

right_shld  = keypoints_pixel[6]
right_elb   = keypoints_pixel[8]
right_wrst  = keypoints_pixel[10]

left_hip    = keypoints_pixel[11]
left_knee   = keypoints_pixel[13]
left_ankl   = keypoints_pixel[15]

right_hip   = keypoints_pixel[12]
right_knee  = keypoints_pixel[14]
right_ankl  = keypoints_pixel[16]


nose_n        = keypoints_norm[0]

left_eye_n    = keypoints_norm[1]
left_ear_n    = keypoints_norm[3]

right_eye_n   = keypoints_norm[2]
right_ear_n   = keypoints_norm[4]

left_shld_n   = keypoints_norm[5]
left_elb_n    = keypoints_norm[7]
left_wrst_n   = keypoints_norm[9]

right_shld_n  = keypoints_norm[6]
right_elb_n   = keypoints_norm[8]
right_wrst_n  = keypoints_norm[10]

left_hip_n    = keypoints_norm[11]
left_knee_n   = keypoints_norm[13]
left_ankl_n   = keypoints_norm[15]

right_hip_n   = keypoints_norm[12]
right_knee_n  = keypoints_norm[14]
right_ankl_n  = keypoints_norm[16]

print('------------------------------------------------------')
angle1 = compute_angle(left_shld, left_elb, left_wrst)
print(f"Unghi mana stanga: {angle1} grade")

angle2 = compute_angle(right_shld, right_elb, right_wrst)
print(f"Unghi mana dreapta: {angle2} grade")

angle3 = compute_angle(left_hip, left_knee, left_ankl)
print(f"Unghi picior stang: {angle3} grade")

angle4 = compute_angle(right_hip, right_knee, right_ankl)
print(f"Unghi picior drept: {angle4} grade")

if angle1>140 or angle2>140 or angle3>140 or angle4>140 or angle3<100 or angle5<100:
    print("Flexarea membrelor este incorecta, risc de dezechilibru!")
else:
    print("Postura este suficient de flexata.")
print('------------------------------------------------------')
angle5 = compute_angle(nose, left_hip, left_ankl)
print(f"Unghi aplecare 1 (stanga): {angle5} grade")

angle6 = compute_angle(nose, right_hip, right_ankl)
print(f"Unghi aplecare 2 (dreapta): {angle6} grade")

if angle5>145 or angle6>145:
    print("Inclinare periculoasa!")
else:
    print("Echilibru corect.")

print('------------------------------------------------------')
print(f"Glezna stanga norm: {left_ankl_n}")
print(f"Glezna dreapta norm: {right_ankl_n}")

dist = calculate_distance(left_ankl_n, right_ankl_n)
print(f"Distanta dintre cele doua glezne este: {dist}")

if dist>0.12:
    print("Schiurile sunt prea departate!")
else:
    print("Schiurile sunt suficient de paralele.")

print('------------------------------------------------------')
#palmele sa nu se duca mai sus decat umerii
if min(left_wrst_n[1],right_wrst_n[1]) <= max(left_shld_n[1],right_shld_n[1]):
    print("Mainile sunt prea sus!")
else:
    print("Elevatie corespunzatoare a mainilor.")
    
print('------------------------------------------------------')
#fundul sa nu fie mai jos decat genunchii
if min(left_knee_n[1],right_knee_n[1]) <= max(left_hip_n[1],right_hip_n[1]):
    print("Posteriorul este prea jos, risc de dezechilibru!")
else:
    print("Elevatie corespunzatoare a posteriorului.")

print('------------------------------------------------------')
#gleznele mai sus decat orice in afara de palme, atunci = cazatura
if min(y for x, y in keypoints_norm)<left_ankl_n[1] or min(y for x, y in keypoints_norm)<right_ankl_n[1] :
    print("Ranire detectata!")



#---------------------------------------------------------------------------------------------------------------------------#


#EXTRA LUCRURI PENTRU VIZUALIZARE
# print('------------------------------------------------------')
# print(keypoints.data)

# print('------------------------------------------------------')
# print(keypoints.xy)

# print('------------------------------------------------------')
# print(keypoints.xyn)

# #EXTRA LUCRURI PENTRU VIZUALIZARE
# print('------------------------------------------------------')
# print(keypoints_pixel.data)