import argparse
import json
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import Resize, ConvertImageDtype, Normalize
from functools import partial
import myutils
import Mymodel
import numpy as np
import albumentations as A
from mylabels import trainId2label
import matplotlib.pyplot as plt

def visualize_map(image, segmentation_map):
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    for trainId, label in trainId2label.items():
        color_seg[segmentation_map == trainId, :] = label.color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    #plt.show()
    plt.savefig("inference.png")

    return

def preprocess_image(test_image, device):

    #ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    #ADE_STD = np.array([58.395, 57.120, 57.375]) / 255 
    ADE_MEAN = [0.485, 0.456, 0.406]
    ADE_STD = [0.229, 0.224, 0.225]    

    val_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),

    ])

    image=np.array(test_image)
    pixel_values = val_transform(image=image)["image"]
    pixel_values = torch.tensor(pixel_values)
    pixel_values = pixel_values.permute(2,0,1).unsqueeze(0)
    
    return pixel_values

def main():

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize the model
    model = Mymodel.load_model(device)
    # Load model weights
    model.load_state_dict(torch.load(args.model_weights_path, weights_only=True))

    # Start the verification mode of the model.
    model.eval()

    test_image = Image.open(args.image_path)
    input_img = preprocess_image(test_image, device)

    with torch.no_grad():
        outputs = model(input_img.to(device))

    upsampled_logits = torch.nn.functional.interpolate(outputs.logits, 
                                                       size=test_image.size[::-1], 
                                                       mode="bilinear", align_corners=False)
    predicted_map = upsampled_logits.argmax(dim=1) 

    visualize_map(test_image, predicted_map.squeeze().cpu())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights_path", type=str, default="./myoutput/50_best_model.pth")
    parser.add_argument("--image_path", type=str, default="/home/cap6411.student1/CVsystem/assignment/hw8/cityscapes/val/img/val102.png")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    main()
