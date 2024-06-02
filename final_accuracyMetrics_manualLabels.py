import os
import json
from tkinter import Tk, Label, Button, Checkbutton, IntVar
from PIL import Image, ImageTk

# Directories
base_dir = 'C:/Users/maria/Desktop/Master/Dizertatie/injuryski_dataset/injuryski_dataset'
image_dir = os.path.join(base_dir, 'images')
annotation_dir = os.path.join(base_dir, 'annotations')
output_file = os.path.join(base_dir, 'poseLabels.json')

# List to hold images that are already annotated
annotated_images = []
labels = []

def load_and_filter_annotations():
    for root, _, files in os.walk(annotation_dir):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(root, filename), 'r') as file:
                    annotation = json.load(file)
                    for image_info in annotation['images']:
                        if image_info['is_labeled']:
                            image_path = os.path.join(base_dir, image_info['file_name'])
                            annotated_images.append((image_path, image_info))

class LabelingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Labeling App")

        self.image_label = Label(master)
        self.image_label.pack()

        self.fall_detected_var = IntVar()
        self.loss_of_balance_var = IntVar()

        self.fall_checkbox = Checkbutton(master, text="Fall detected", variable=self.fall_detected_var)
        self.fall_checkbox.pack()

        self.balance_checkbox = Checkbutton(master, text="Loss of balance", variable=self.loss_of_balance_var)
        self.balance_checkbox.pack()

        self.next_button = Button(master, text="Next", command=self.next_image)
        self.next_button.pack()

        self.save_button = Button(master, text="Save", command=self.save_label)
        self.save_button.pack()

        self.image_index = -1
        self.current_image = None
        self.current_annotation = None

        load_and_filter_annotations()
        self.next_image()

    def next_image(self):
        self.image_index += 1
        if self.image_index < len(annotated_images):
            image_path, annotation = annotated_images[self.image_index]
            self.current_image = Image.open(image_path)
            self.current_image.thumbnail((500, 500))
            self.current_annotation = annotation
            self.display_image()
        else:
            self.image_label.config(text="No more images to label")
            with open(output_file, 'w') as file:
                json.dump(labels, file, indent=4)

    def display_image(self):
        img = ImageTk.PhotoImage(self.current_image)
        self.image_label.config(image=img)
        self.image_label.image = img

    def save_label(self):
        fall_detected = self.fall_detected_var.get() == 1
        loss_of_balance = self.loss_of_balance_var.get() == 1

        label_data = {
            'image_path': annotated_images[self.image_index][0],
            'fall_detected': fall_detected,
            'loss_of_balance': loss_of_balance
        }

        labels.append(label_data)

        self.next_image()

if __name__ == "__main__":
    root = Tk()
    app = LabelingApp(root)
    root.mainloop()
