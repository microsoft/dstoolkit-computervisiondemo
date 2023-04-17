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

def ensure_divisible(num, denom):
    """Ensures input number is divisable by denom, with a goal that both width and height will be even. 
    This is so when downsampling/ upsampling in unet, each skpped connection matches up 
    when concating in the upscale section ie skipped connection and convoluted img match dims."""
    if num % denom == 0:
        return num
    else:
        return_num = num
        while return_num % denom != 0:
            return_num += 1
        return return_num

def resize_with_padding(img, expected_size, colour):
    """Resizes images with padding and scaling to expected sizes using PIL."""
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0] # get 
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill = colour)

def binary_mask(pil_img):
    """Takes image path as input and returns binary (0, or 255 in RGB) image."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = np.array(img)
    img_copy = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 2:
                img_copy[i, j] = 0
            else:
                img_copy[i, j] = 1
    pil_img_rtn = Image.fromarray(img_copy)
    return pil_img_rtn

#### below code gets avg width and height 
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
####

# get mean width/ height to rescale to this
scale_down = 2
num_features_UNET = 4
denom = num_features_UNET**2 # each feature must be squared as this is the downsample rate, so to match with upsample in UNET. Ie 4 features, needs to be able to be halved 4 times to matchthe up sample.
mean_width = ensure_divisible(int(np.mean(width_list)/scale_down), denom)
mean_height = ensure_divisible(int(np.mean(height_list)/scale_down), denom)

size = max(mean_width, mean_height) # in experimentation ive found better to do squares (with padding) so that convolution can be applied to both axis
mean_width, mean_height=size,size
print(f"\n\nMean width and height: {mean_width, mean_height}\n\n")

# Once got mean width and height, resize img labels and training data to mean (taking mean is less computationally expensive overall as it represents mid point of data)
for filename_full in os.listdir(IMAGE_DATASET_PATH): # iterate again and scale to same size. 
    file_path = os.path.join(IMAGE_DATASET_PATH, filename_full)
    if os.path.isfile(file_path):
        filename_split = filename_full.split(".")
        filename, ext = filename_split[0], filename_split[1]
        if filename not in faulty_file_names: # only get non-errors
            try:# try except due to some issues with libraries being used. 
                # X images resize
                img_feature = Image.open(file_path)
                resize_img_feature = resize_with_padding(img_feature, (mean_width, mean_height), (255, 255, 255))  
                              
                # y labels resize and binary
                label_file_path = os.path.join(MASK_DATASET_PATH, filename + ".png")
                img_label = Image.open(label_file_path)
                resize_img_label = resize_with_padding(img_label, (mean_width, mean_height), 2)
                pil_binary_mask = binary_mask(resize_img_label)
                
                # Save
                resize_img_feature.save(os.path.join(RESIZE_FEATURES_PATH, filename + ".png"))
                pil_binary_mask.save(os.path.join(RESIZE_LABELS_PATH, filename + ".png"))
            except: 
                pass
            
print(f"Difference in features and labels: {len(os.listdir(RESIZE_FEATURES_PATH)) - len(os.listdir(RESIZE_LABELS_PATH))}")


# import cv2
# import os
# import numpy as np
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt

# WORKING_PATH = os.getcwd() # called in notebook for working dir
# DATASET_PATH = os.path.join(WORKING_PATH, "data")
# BASE_OUTPUT = os.path.join(WORKING_PATH, "outputs")
# IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "raw", "images")
# MASK_DATASET_PATH = os.path.join(DATASET_PATH, "raw", "annotations", "trimaps")
# RESIZE_FEATURES_PATH = os.path.join(DATASET_PATH, "preproccessed", "features")
# RESIZE_LABELS_PATH = os.path.join(DATASET_PATH, "preproccessed", "labels")

# def show_image_mask(img, mask, cmap='gray'): 
#     fig = plt.figure(figsize=(5,5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img, cmap=cmap)
#     plt.axis('off')
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask, cmap=cmap)
#     plt.axis('off')

# name = "yorkshire_terrier_197.png"
# image = cv2.imread(os.path.join(RESIZE_FEATURES_PATH,name), cv2.IMREAD_UNCHANGED)
# mask = cv2.imread(os.path.join(RESIZE_LABELS_PATH, name), cv2.IMREAD_UNCHANGED)
# show_image_mask(image, mask, cmap='gray')