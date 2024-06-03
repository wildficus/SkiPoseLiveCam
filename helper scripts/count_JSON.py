import json

# Path to the JSON file
json_file_path = 'C:/Users/maria/Desktop/Master/Dizertatie/injuryski_dataset/injuryski_dataset/poseLabels.json'  # Replace with the actual path to your JSON file

# Load JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Count the number of images
num_images = len(data)

# Print the number of images
print(f"Number of images in the JSON file: {num_images}")
