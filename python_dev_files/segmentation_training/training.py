# documentation/ references 
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from tqdm import tqdm # library to show progress

import torch
from torch.utils.data import Dataset
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import gc

# clear memory
torch.cuda.empty_cache()
gc.collect()

WORKING_PATH = os.getcwd() # called in notebook for working dir
DATASET_PATH = os.path.join(WORKING_PATH, "data")
RESIZE_FEATURES_PATH = os.path.join(DATASET_PATH, "preproccessed", "features")
RESIZE_LABELS_PATH = os.path.join(DATASET_PATH, "preproccessed", "labels")
BASE_OUTPUT = os.path.join(WORKING_PATH, "outputs")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
MODEL_PATH = os.path.join(BASE_OUTPUT, "model.pkl")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes, and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 3 # outline, BG and body
NUM_LEVELS = 3 # unet levels
INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT =  mean_width, mean_height

# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 32
THRESHOLD = 0.5 # define threshold to filter weak predictions

# define dataset loader
class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		self.imagePaths = imagePaths # features
		self.maskPaths = maskPaths # labels/ masks
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return (image, mask)

class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs

class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		# return the final decoder output
		return x
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		return encFeatures 		# return the cropped features

class UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64),
		 decChannels=(64, 32, 16),
		 nbClasses=NUM_CLASSES, retainDim=True,
		 outSize=(INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH)):
		super().__init__()
		# initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		# initialize the regression head and store the class variables
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
  
	def forward(self, x):
		# grab the features from the encoder
		encFeatures = self.encoder(x)
		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		decFeatures = self.decoder(encFeatures[::-1][0],
			encFeatures[::-1][1:])
		# pass the decoder features through the regression head to
		# obtain the segmentation mask
		map = self.head(decFeatures)
		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retainDim:
			map = F.interpolate(map, self.outSize)
		# return the segmentation map
		return map

# load the image and mask filepaths in a sorted manner
imagePaths = [RESIZE_FEATURES_PATH + "/" + filename for filename in sorted(list(os.listdir(RESIZE_FEATURES_PATH)))]
maskPaths = [RESIZE_LABELS_PATH + "/" + filename for filename in sorted(list(os.listdir(RESIZE_LABELS_PATH)))]

split = train_test_split(imagePaths, maskPaths,
	test_size=0.2, random_state=37)
(trainImages, testImages) = split[:2] # unpack the data split
(trainMasks, testMasks) = split[2:]
print("[INFO] saving testing image paths...")
f = open(TEST_PATHS, "w") 
f.write("\n".join(testImages)) # write the testing image paths to disk so that we can use then when evaluating/testing our model
f.close()

transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]) # define transformations
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)

# initialize UNet model
unet = UNet().to(DEVICE)

lossFunc = BCEWithLogitsLoss() # initialize loss function and optimizer
opt = Adam(unet.parameters(), lr=INIT_LR)
trainSteps = len(trainDS) // BATCH_SIZE
testSteps = len(testDS) // BATCH_SIZE
train_hist = {"train_loss": [], "test_loss": []} # initialize a dictionary to store training history

################
# Training Loop
################
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(NUM_EPOCHS)): # loop over epochs
	unet.train() # set the model in training mode
	totalTrainLoss, totalTestLoss = 0, 0 # initialize the total training and validation loss

	for (i, (x, y)) in enumerate(trainLoader): 	# loop over the training set
		# send the input to the device
		(x, y) = (x.to(DEVICE), y.to(DEVICE))
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward() # after each batch
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
  
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(DEVICE), y.to(DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training history
	train_hist["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	train_hist["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
 
endTime = time.time()# display the total time needed to perform the training
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))