import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch.nn as nn
import gc

# clear memory
torch.cuda.empty_cache()
gc.collect()

WORKING_PATH = os.getcwd() # called in notebook for working dir
# Inputs
RESIZE_FEATURES_PATH = os.path.join(WORKING_PATH, "data", "preproccessed", "features")
RESIZE_LABELS_PATH = os.path.join(WORKING_PATH, "data", "preproccessed", "labels")
# Outputs
BASE_OUTPUT = os.path.join(WORKING_PATH, "outputs")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
MODEL_PATH = os.path.join(BASE_OUTPUT, "model.pkl")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define dataset loader
class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths):
		self.imagePaths = imagePaths # features
		self.maskPaths = maskPaths # labels/ masks
  
	def __len__(self):
		return len(self.imagePaths) # return the number of total samples contained in the dataset

	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		maskPath = self.maskPaths[idx]
		data = cv2.imread(imagePath, cv2.COLOR_BGRA2BGR)
		label = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
		
		return (torch.from_numpy(data).float(), torch.from_numpy(label).float())

class DoubleConv(nn.Module):
    """https://www.youtube.com/watch?v=IHq1t7NxS8k"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__() # what does super do?
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # set bias to false as using batchnorm (cancels bias)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module): # means inherit from nn.module
    def __init__(self, in_channels=3, out_channels=1, features = [64,128,256,512]): # channels refer to channel so rgb 
        """init creates arcitecture for the class instance. """
        super(UNet, self,).__init__()
        self.ups = nn.ModuleList() # lists that store convolution layers, module list 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (kernel_size=2, stride=2) = divisible by 2
        
        for feature in features: # down blocks of UNET, go through features 
            self.downs.append(DoubleConv(in_channels = in_channels, out_channels = feature)) # add layer to UNET
            in_channels = feature # set new in channels as last our channel

        for feature in reversed(features): # up blocks
            self.ups.append(
                nn.ConvTranspose2d(in_channels = feature*2, out_channels = feature, kernel_size=2, stride=2)) # up sample 
            self.ups.append(
                DoubleConv(in_channels = feature*2, out_channels = feature)) # 2 convolutions

        self.bottleneck = DoubleConv(in_channels = features[-1], out_channels = features[-1] * 2) # bottleneck. Last feature in list in and then last feqature x 2 to go out

        # final conv layer
        self.finalConv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size = 1) # change number of channels 

    def forward(self, x):
        skip_connections = [] # store all the skip connections
        for down in self.downs: # down blocks
            x = down(x) # doubleconv
            skip_connections.append(x) # used to take across the U
            x = self.pool(x) # down sampling

        x = self.bottleneck(x) # bottleneck and double conv
        skip_connections = skip_connections[::-1] # inverse skip connections (as we will use them in reverse order that we added them)
        
        for idx in range(0,len(self.ups),2):  # loop through every 2 ups (as each up step has the ConvTranspose and the DoubleConv)  
            x = self.ups[idx](x) # convtranspose (up sampling)
            skipped_connection = skip_connections[idx//2] # get the skipped connection
            concat_skip = torch.cat((skipped_connection,x),dim=1) # Dim 1 is channel dim. Concatenate upsampled activation maps with skipped connection
            x = self.ups[idx+1](concat_skip) # doubleconv with concatenated activation maps

        x = self.finalConv(x) # do final convolution to get mask
        return x
    
def convertTime(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)
    
# load the image and mask filepaths in a sorted manner
imagePaths = [RESIZE_FEATURES_PATH + "/" + filename for filename in sorted(list(os.listdir(RESIZE_FEATURES_PATH)))]
maskPaths = [RESIZE_LABELS_PATH + "/" + filename for filename in sorted(list(os.listdir(RESIZE_LABELS_PATH)))]

split = train_test_split(imagePaths, maskPaths, test_size=0.15, random_state=37)
(trainImages, testImages) = split[:2] # unpack the data split
(trainMasks, testMasks) = split[2:]

with open(TEST_PATHS, "w") as f:
	f.write("\n".join(testImages)) # write the testing image paths to disk so that we can use then when evaluating/testing our model
	f.close()
print("[INFO] saving testing image paths...")

trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks)
print(f"[INFO] found {len(trainDS)} examples in the training set.\n[INFO] found {len(testDS)} examples in the test set.")

width = 224
height = 224
batch_size = 5
num_epochs = 2
lr_INIT=0.001

number_batches = int(np.ceil(len(trainDS) / batch_size))
model = UNet()
criterion = BCEWithLogitsLoss() # we can use binary cross entropy with logit OR use sigmoid on last step and use BCELoss
optimizer = Adam(model.parameters(), lr=lr_INIT)

highest_loss = np.inf # initilise as high
best_epoch = 0
for epoch in range(num_epochs):
  start_time = time.time()
  epoch_loss = 0.0
  
  # forward
  for batch_num, batch_idx in enumerate(range(0,len(trainDS)-1,batch_size)):   # for each batch, i = 0,5,10 etc (for batchsize 5)
    inputs = torch.tensor([])
    masks = torch.tensor([])
    for image_idx in range(0,batch_size): # for image in batch, j = 0,1,2,3,4 (for batchsize 5)
      input_obj, mask_obj = trainDS[batch_idx+image_idx][0], trainDS[batch_idx+image_idx][1]  # get image and mask at index
      input = torch.reshape(input_obj,(3,width,height))
      mask = torch.reshape(mask_obj,(1,width,height))
      inputs = torch.cat((inputs,input)) # concatenate image and mask to lists of images and masks
      masks = torch.cat((masks,mask))

    # backward
    optimizer.zero_grad()
    inputs = torch.reshape(inputs,(batch_size, 3, width, height))     # get predictions and predicted mask
    preds = model(inputs)

    # loss    
    masksL = masks.type(torch.FloatTensor) # put masks form that criterion wants
    preds = preds.reshape(5, width, height)
    loss = criterion(preds, masksL) # get loss
    loss.backward() # backpropogate
    optimizer.step()

    batch_endTime = time.time()
    batch_timeDelta = convertTime(batch_endTime - start_time)
    epoch_loss += loss.item()
    print(f"[INFO] Epoch:{epoch+1}/{num_epochs} | Batch:{batch_num+1}/{number_batches} | Time Elapsed: {batch_timeDelta} | Batch Loss:{loss.item()}")
  
  epoch_endTime = time.time()
  epoch_timeDelta = convertTime(epoch_endTime - start_time)
  print(f"\n[INFO] Epoch:{epoch+1}/{num_epochs} | Time Elapsed: {epoch_timeDelta} | Epoch Loss:{epoch_loss}\n")

  # if (sumDiceScore > highest_dice): # save after every epoch if better
  #   highest_dice = sumDiceScore
  #   best_epoch = epoch
  #   torch.save(model.state_dict(), MODEL_PATH)
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch.nn as nn
import gc

# clear memory
torch.cuda.empty_cache()
gc.collect()

WORKING_PATH = os.getcwd() # called in notebook for working dir
# Inputs
RESIZE_FEATURES_PATH = os.path.join(WORKING_PATH, "data", "preproccessed", "features")
RESIZE_LABELS_PATH = os.path.join(WORKING_PATH, "data", "preproccessed", "labels")
# Outputs
BASE_OUTPUT = os.path.join(WORKING_PATH, "outputs")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
MODEL_PATH = os.path.join(BASE_OUTPUT, "model.pkl")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define dataset loader
class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths):
		self.imagePaths = imagePaths # features
		self.maskPaths = maskPaths # labels/ masks
  
	def __len__(self):
		return len(self.imagePaths) # return the number of total samples contained in the dataset

	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		maskPath = self.maskPaths[idx]
		data = cv2.imread(imagePath, cv2.COLOR_BGRA2BGR)
		label = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
		
		return (torch.from_numpy(data).float(), torch.from_numpy(label).float())

class DoubleConv(nn.Module):
    """https://www.youtube.com/watch?v=IHq1t7NxS8k"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__() # what does super do?
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # set bias to false as using batchnorm (cancels bias)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module): # means inherit from nn.module
    def __init__(self, in_channels=3, out_channels=1, features = [64,128,256,512]): # channels refer to channel so rgb 
        """init creates arcitecture for the class instance. """
        super(UNet, self,).__init__()
        self.ups = nn.ModuleList() # lists that store convolution layers, module list 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (kernel_size=2, stride=2) = divisible by 2
        
        for feature in features: # down blocks of UNET, go through features 
            self.downs.append(DoubleConv(in_channels = in_channels, out_channels = feature)) # add layer to UNET
            in_channels = feature # set new in channels as last our channel

        for feature in reversed(features): # up blocks
            self.ups.append(
                nn.ConvTranspose2d(in_channels = feature*2, out_channels = feature, kernel_size=2, stride=2)) # up sample 
            self.ups.append(
                DoubleConv(in_channels = feature*2, out_channels = feature)) # 2 convolutions

        self.bottleneck = DoubleConv(in_channels = features[-1], out_channels = features[-1] * 2) # bottleneck. Last feature in list in and then last feqature x 2 to go out

        # final conv layer
        self.finalConv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size = 1) # change number of channels 

    def forward(self, x):
        skip_connections = [] # store all the skip connections
        for down in self.downs: # down blocks
            x = down(x) # doubleconv
            skip_connections.append(x) # used to take across the U
            x = self.pool(x) # down sampling

        x = self.bottleneck(x) # bottleneck and double conv
        skip_connections = skip_connections[::-1] # inverse skip connections (as we will use them in reverse order that we added them)
        
        for idx in range(0,len(self.ups),2):  # loop through every 2 ups (as each up step has the ConvTranspose and the DoubleConv)  
            x = self.ups[idx](x) # convtranspose (up sampling)
            skipped_connection = skip_connections[idx//2] # get the skipped connection
            concat_skip = torch.cat((skipped_connection,x),dim=1) # Dim 1 is channel dim. Concatenate upsampled activation maps with skipped connection
            x = self.ups[idx+1](concat_skip) # doubleconv with concatenated activation maps

        x = self.finalConv(x) # do final convolution to get mask
        return x
    
def convertTime(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)
    
# load the image and mask filepaths in a sorted manner
imagePaths = [RESIZE_FEATURES_PATH + "/" + filename for filename in sorted(list(os.listdir(RESIZE_FEATURES_PATH)))]
maskPaths = [RESIZE_LABELS_PATH + "/" + filename for filename in sorted(list(os.listdir(RESIZE_LABELS_PATH)))]

split = train_test_split(imagePaths, maskPaths, test_size=0.15, random_state=37)
(trainImages, testImages) = split[:2] # unpack the data split
(trainMasks, testMasks) = split[2:]

with open(TEST_PATHS, "w") as f:
	f.write("\n".join(testImages)) # write the testing image paths to disk so that we can use then when evaluating/testing our model
	f.close()
print("[INFO] saving testing image paths...")

trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks)
print(f"[INFO] found {len(trainDS)} examples in the training set.\n[INFO] found {len(testDS)} examples in the test set.")

width = 224
height = 224
batch_size = 5
num_epochs = 2
lr_INIT=0.001

number_batches = int(np.ceil(len(trainDS) / batch_size))
model = UNet()
criterion = BCEWithLogitsLoss() # we can use binary cross entropy with logit OR use sigmoid on last step and use BCELoss
optimizer = Adam(model.parameters(), lr=lr_INIT)

highest_loss = np.inf # initilise as high
best_epoch = 0
for epoch in range(num_epochs):
  start_time = time.time()
  epoch_loss = 0.0
  
  # forward
  for batch_num, batch_idx in enumerate(range(0,len(trainDS)-1,batch_size)):   # for each batch, i = 0,5,10 etc (for batchsize 5)
    inputs = torch.tensor([])
    masks = torch.tensor([])
    for image_idx in range(0,batch_size): # for image in batch, j = 0,1,2,3,4 (for batchsize 5)
      input_obj, mask_obj = trainDS[batch_idx+image_idx][0], trainDS[batch_idx+image_idx][1]  # get image and mask at index
      input = torch.reshape(input_obj,(3,width,height))
      mask = torch.reshape(mask_obj,(1,width,height))
      inputs = torch.cat((inputs,input)) # concatenate image and mask to lists of images and masks
      masks = torch.cat((masks,mask))

    # backward
    optimizer.zero_grad()
    inputs = torch.reshape(inputs,(batch_size, 3, width, height))     # get predictions and predicted mask
    preds = model(inputs)

    # loss    
    masksL = masks.type(torch.FloatTensor) # put masks form that criterion wants
    preds = preds.reshape(5, width, height)
    loss = criterion(preds, masksL) # get loss
    loss.backward() # backpropogate
    optimizer.step()

    batch_endTime = time.time()
    batch_timeDelta = convertTime(batch_endTime - start_time)
    epoch_loss += loss.item()
    print(f"[INFO] Epoch:{epoch+1}/{num_epochs} | Batch:{batch_num+1}/{number_batches} | Time Elapsed: {batch_timeDelta} | Batch Loss:{loss.item()}")
  
  epoch_endTime = time.time()
  epoch_timeDelta = convertTime(epoch_endTime - start_time)
  print(f"\n[INFO] Epoch:{epoch+1}/{num_epochs} | Time Elapsed: {epoch_timeDelta} | Epoch Loss:{epoch_loss}\n")

  # if (sumDiceScore > highest_dice): # save after every epoch if better
  #   highest_dice = sumDiceScore
  #   best_epoch = epoch
  #   torch.save(model.state_dict(), MODEL_PATH)
