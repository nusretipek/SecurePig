# Import libraries

## General
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.ndimage import label

## PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as pth_transforms

## Transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Parameters
transformInference = A.Compose([A.ToFloat(), ToTensorV2()])
imgSizeInitial = (768, 768)
imgSize = (640, 352)
imgSizeOut = (640, 360) 
xRange = (80, 540)
numPoints = 25000

# Unet architeture
class UNet(nn.Module):
    def __init__(self, input_channels, device):
        super(UNet, self).__init__()
        self.device = device
        self.convP1 = 32
        self.convP2 = 64
        self.convP3 = 128
        self.convP4 = 256       
        self.convP5 = 512
        
        self.conv1 = self.conv_block(input_channels, self.convP1)
        self.conv2 = self.conv_block(self.convP1, self.convP2)
        self.conv3 = self.conv_block(self.convP2, self.convP3)
        self.conv4 = self.conv_block(self.convP3, self.convP4)
        self.conv5 = self.conv_block(self.convP4, self.convP5)
        
        self.convStride1 = nn.Conv2d(self.convP1, self.convP1, kernel_size=3, padding=1, stride=(2,2))
        self.convStride2 = nn.Conv2d(self.convP2, self.convP2, kernel_size=3, padding=1, stride=(2,2))
        self.convStride3 = nn.Conv2d(self.convP3, self.convP3, kernel_size=3, padding=1, stride=(2,2))
        self.convStride4 = nn.Conv2d(self.convP4, self.convP4, kernel_size=3, padding=1, stride=(2,2))

        self.convsingle1 = self.conv_block_single(self.convP5, self.convP4)
        self.convsingle2 = self.conv_block_single(self.convP4, self.convP3)
        self.convsingle3 = self.conv_block_single(self.convP3, self.convP2)
        self.convsingle4 = self.conv_block_single(self.convP2, self.convP1)
        
        self.upconv1 = self.conv_block2(self.convP5, self.convP4)
        self.upconv2 = self.conv_block2(self.convP4, self.convP3)
        self.upconv3 = self.conv_block2(self.convP3, self.convP2)
        self.upconv4 = self.conv_block2(self.convP2, self.convP1)        
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.ConvTranspose2d(self.convP5, self.convP5, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(self.convP4, self.convP4, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(self.convP3, self.convP3, kernel_size=2, stride=2)
        self.upsample5 = nn.ConvTranspose2d(self.convP2, self.convP2, kernel_size=2, stride=2)

        self.dropout = nn.Dropout2d(p=0.2)
        self.dropoutEncoding = nn.Dropout2d(p=0.05)

        self.final_conv1 = nn.Conv2d(self.convP1, 2, kernel_size=1, padding='same')
        self.final_conv2 = nn.Conv2d(2, 1, kernel_size=1, padding='valid')

        self.initialize_weights() # Initialize weights with He normal initialization

        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        ).to(self.device)
        
    def conv_block2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ).to(self.device)

    def conv_block_single(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        ).to(self.device)

    def forward(self, x):
        conv1 = self.conv1(x)
        drop1 = self.dropoutEncoding(conv1)
        sdrop1 = self.convStride1(drop1)

        conv2 = self.conv2(sdrop1)
        drop2 = self.dropoutEncoding(conv2)
        sdrop2 = self.convStride2(drop2)

        conv3 = self.conv3(sdrop2)
        drop3 = self.dropoutEncoding(conv3)
        sdrop3 = self.convStride3(drop3)

        conv4 = self.conv4(sdrop3)
        drop4 = self.dropout(conv4)
        sdrop4 = self.convStride4(drop4)

        conv5 = self.conv5(sdrop4)
        drop5 = self.dropout(conv5)
        
        up6 = self.upsample(drop5)
        up6 = self.convsingle1(up6)
        merge6 = torch.cat([drop4, up6], dim=1)
        conv6 = self.upconv1(merge6)

        up7 = self.upsample(conv6)
        up7 = self.convsingle2(up7)
        merge7 = torch.cat([drop3, up7], dim=1)
        conv7 = self.upconv2(merge7)
        
        up8 = self.upsample(conv7)
        up8 = self.convsingle3(up8)
        merge8 = torch.cat([drop2, up8], dim=1)
        conv8 = self.upconv3(merge8)
        
        up9 = self.upsample(conv8)
        up9 = self.convsingle4(up9)
        merge9 = torch.cat([drop1, up9], dim=1)
        conv9 = self.upconv4(merge9)        

        final_conv = self.final_conv1(conv9)
        final_conv = self.final_conv2(final_conv)
        output = torch.sigmoid(final_conv)
        return output

# Single image inference
def getPrediction(model, img, device):
    with torch.no_grad():
        predicted_masks = model(img.unsqueeze(0).to(device)).cpu().squeeze()
        predicted_masks = F.interpolate(predicted_masks.unsqueeze(0).unsqueeze(0), size=(imgSizeOut[1], imgSizeOut[0]), mode='nearest').squeeze()
        predicted_masks_numpy = predicted_masks.cpu().numpy()
    return predicted_masks, predicted_masks_numpy

def filterTensor(indices, imageSize=(imgSizeOut[1], imgSizeOut[0]), minArea=1):
    pointTensor = torch.zeros(imageSize, dtype=torch.int).flatten()
    pointTensor[indices] = 1
    pointTensor = pointTensor.reshape(imageSize).numpy()
    pointTensor = maximum_filter(pointTensor, size=(5, 5))
    labeled_array, num_features = label(pointTensor)
    
    filtered_mask = np.zeros_like(pointTensor)
    for region_label in range(1, num_features + 1):
        region_size = np.sum(labeled_array == region_label)
        if region_size >= minArea:
            filtered_mask[labeled_array == region_label] = 1
    filtered_mask = maximum_filter(filtered_mask, size=(1, 1))        
    return filtered_mask

def overlay_mask_on_image(image_path, mask_array, save_path):
    image = Image.open(image_path).convert('RGB').resize(imgSize)
    mask_rgb = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
    mask_rgb[mask_array == 1] = [0, 255, 0, int(255 * 0.5)]
    mask_image = Image.fromarray(mask_rgb, 'RGB')
    overlaid_image = Image.alpha_composite(image.convert('RGB'), mask_image)
    overlaid_image.save(save_path)
    return overlaid_image

# Load model
def getModel(checkpointPath, device):
    model = UNet(input_channels=3, device=device)
    model.load_state_dict(torch.load(checkpointPath, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model

# Inference
def getFeatures(model, imgPath, device):
    # Get prediction from Unet
    image = Image.open(imgPath).convert('RGB').resize(imgSizeInitial)
    augmented = transformInference(image=np.array(image))
    image = augmented['image']
    predicted_masks, predicted_masks_numpy = getPrediction(model, image, device)
    predicted_masks_numpy[predicted_masks_numpy > 0.5] = 1
    predicted_masks_numpy[predicted_masks_numpy < 0.5] = 0
    return predicted_masks.tolist()
    
    # Copy Tensors
    thresoldTensor = predicted_masks.clone()
    tensor = predicted_masks_numpy.copy()
    
    # Get Threshold value
    thresoldTensor[thresoldTensor < torch.quantile(thresoldTensor, 0.9)] = 0
    thresoldTensor[thresoldTensor < -1] = 0
    threshold = int(thresoldTensor.sum().item())
    threshold = min(threshold, numPoints)
    threshold = max(threshold, 0)
    #tensor[:, :xRange[0]] = 0
    #tensor[:, xRange[1]:] = 0
    
    # Flatten the tensor and get indices above threshold elements
    flat_tensor = tensor.flatten()
    top_indices = np.argsort(flat_tensor)[-threshold:] 
    
    # Convert the flat indices back to 2D coordinates
    x_coords, y_coords = np.unravel_index(top_indices, tensor.shape)
    top_20_values = flat_tensor[top_indices]
    
    # Filter the image
    filteredTensor = filterTensor(top_indices)
    #pointMask = np.array(Image.fromarray(filteredTensor).resize(imgSizeOut)).tolist()
    pointMask = filteredTensor.tolist()
    return pointMask#predicted_masks.tolist()






