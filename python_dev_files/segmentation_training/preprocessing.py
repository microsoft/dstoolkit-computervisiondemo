# preprocessing 
import cv2
import os
import numpy as np
from PIL import Image, ImageOps

WORKING_PATH = os.getcwd() # called in notebook for working dir
DATASET_PATH = os.path.join(WORKING_PATH, "data")
BASE_OUTPUT = os.path.join(WORKING_PATH, "outputs")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "raw", "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "raw", "annotations", "trimaps")
RESIZE_FEATURES_PATH = os.path.join(DATASET_PATH, "preproccessed", "features")
RESIZE_LABELS_PATH = os.path.join(DATASET_PATH, "preproccessed", "labels")

def resize_padding(image, expected_size, colour):
    """Resizes images with padding and scaling to expected sizes using PIL."""
    image.thumbnail((expected_size[0], expected_size[1])) # PIL function
    delta_width, delta_height = expected_size[0] - image.size[0], expected_size[1] - image.size[1] # get change in height
    pad_width, pad_height = delta_width // 2, delta_height // 2 # get padding width and height
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height) # wrap padding for PIL function
    return ImageOps.expand(image, padding, fill = colour) # use PIL function to perform padding 

width_list = []
height_list = []
faulty_file_names = []
for filename_full in os.listdir(IMAGE_DATASET_PATH):
    file_path = os.path.join(IMAGE_DATASET_PATH, filename_full)
    if os.path.isfile(file_path):
        filename_split = filename_full.split(".")
        filename, ext = filename_split[0], filename_split[1]
        try:
            img = cv2.imread(file_path)
            image_shape = img.shape
            width, height = image_shape[1], image_shape[0]
            width_list.append(width); height_list.append(height) # check dimensions to find most common 
        except: 
            faulty_file_names.append(filename) # check which cv2 struggles to open so we can drop from images and labels 

# get mean width/ height to rescale to this
mean_width = int(np.mean(width_list))
mean_height = int(np.mean(height_list))
print(f"\n\nMean width and height: {mean_width, mean_height}\n\n")

# Once got mean width and height, resize img labels and training data to mean (taking mean is less computationally expensive overall as it represents mid point of data)
for filename_full in os.listdir(IMAGE_DATASET_PATH): # iterate again and scale to same size. 
    file_path = os.path.join(IMAGE_DATASET_PATH, filename_full)
    if os.path.isfile(file_path):
        filename_split = filename_full.split(".")
        filename, ext = filename_split[0], filename_split[1]
        if filename not in faulty_file_names: # only get non-errors
            try: # try except due to some issues with libraries being used. 
                # X images resize
                img_feature = Image.open(file_path)
                resize_img_feature = resize_with_padding(img_feature, (mean_width, mean_height), (255, 255, 255))  
                              
                # y labels resize
                label_file_path = os.path.join(MASK_DATASET_PATH, filename + ".png")
                img_label = Image.open(label_file_path)
                resize_img_label = resize_with_padding(img_label, (mean_width, mean_height), 2)
                
                # Save
                resize_img_feature.save(os.path.join(RESIZE_FEATURES_PATH, filename + ".png"))
                resize_img_label.save(os.path.join(RESIZE_LABELS_PATH, filename + ".png"))
            except: 
                pass
            
print(f"Difference in features and labels: {len(os.listdir(RESIZE_FEATURES_PATH)) - len(os.listdir(RESIZE_LABELS_PATH))}")