import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
import os
import opendatasets as od
from tqdm import tqdm
import numpy as np
import sys
from pynvml import *
import timeit
import time
import albumentations as A
from PIL import Image
import torch.nn.functional as F

class CityScapesDataset(Dataset):
    def __init__(self, split='train' , datapath='./cityscapes', transform=None):
        self.img_dir = f"{datapath}/{split}/img"
        self.label_dir = f"{datapath}/{split}/mylabel"
        self.img_labels = os.listdir(f"{datapath}/{split}/mylabel")
        self.transform = transform
        self.label_tran = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        org_image = Image.open(img_path)
        org_image = np.array(org_image)
        label_path = os.path.join(self.label_dir, self.img_labels[idx])
        org_label = Image.open(label_path)
        org_label =  np.array(org_label)
        
        if self.transform:
            transformed = self.transform(image=org_image,mask=org_label)
            image, label = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])
            #image = self.transform(org_image)
            #label = self.label_tran(org_label)
            #label = label.long()
            #label = label.squeeze(0)
            #label = F.one_hot(label, num_classes=29)
            #label = label.permute(2, 0, 1)
            #label = label.float()

        # convert to C, H, W
        image = image.permute(2,0,1)
        
        return image, label


def getdata(batch_size, data_path="./cityscapes"):

    isExist = os.path.exists(data_path)
    if isExist==False:
        dataset = 'https://www.kaggle.com/datasets/shuvoalok/cityscapes/data'
        od.download(dataset)
    else:
        print("dataset exist")

    # Load the datasets
    data_dir =  data_path
    
    data_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]) 

    #ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    #ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
    ADE_MEAN = [0.485, 0.456, 0.406]
    ADE_STD = [0.229, 0.224, 0.225]

    train_transform = A.Compose([
        # hadded an issue with an image being too small to crop, PadIfNeeded didn't help...
        # if anyone knows why this is happening I'm happy to read why
        # A.PadIfNeeded(min_height=448, min_width=448),
        # A.RandomResizedCrop(height=448, width=448),
        A.Resize(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])

    val_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),

    ])

    # Split out val dataset from train dataset
    train_dataset = CityScapesDataset(split='train', datapath=data_dir, transform=train_transform)
    n = len(train_dataset)
    n_val = int(0.1 * n)
    val_dataset = torch.utils.data.Subset(train_dataset, range(n_val))
    test_dataset = CityScapesDataset(split='val', datapath=data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("data load successfully!")
    return train_loader, val_loader, test_loader

def calculate_pixel_accuracy(predicted_logits, true_mask, ignore_index=0):
    # Step 1: Get the predicted mask by taking the argmax of the logits along the class dimension
    predicted_mask = torch.argmax(predicted_logits, dim=1)  # Shape: [batch_size, height, width]

    # Step 2: Create a mask to ignore pixels where the true mask is the ignore class (class 0)
    valid_mask = (true_mask != ignore_index)  # Shape: [batch_size, height, width]

    # Step 3: Compare the predicted mask with the true mask, only for valid pixels
    correct_pixels = (predicted_mask == true_mask) & valid_mask
    correct_pixels_count = correct_pixels.sum().item()

    # Step 4: Calculate the total number of valid pixels (i.e., pixels that are not the ignore class)
    total_valid_pixels = valid_mask.sum().item()

    # Step 5: Calculate pixel accuracy (only over valid pixels)
    if total_valid_pixels > 0:
        pixel_accuracy = correct_pixels_count / total_valid_pixels
    else:
        pixel_accuracy = 0.0

    return pixel_accuracy

def train_model(model, train_loader, optimizer, metric, device):
    start_time = time.time()
    model.train()

    nvmlInit()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    running_pic_acc =0.0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
                

        outputs = model(pixel_values=inputs,labels=labels)
        
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predicted = outputs.logits.argmax(dim=1)

            # note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
        
        metrics = metric.compute(num_labels=29,
                                ignore_index=0,
                                reduce_labels=False,
        )

        running_loss += loss.item() * inputs.size(0)
        running_iou += metrics["mean_iou"]
        running_acc += metrics["mean_accuracy"]
        running_pic_acc += calculate_pixel_accuracy(outputs.logits, labels, ignore_index=0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    epoch_iou = running_iou / len(train_loader.dataset)
    epoch_pic_acc = running_pic_acc / len(train_loader.dataset)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Train Loss: {:.4f} Acc: {:.4f} iou: {:.4f} pic_acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, epoch_iou, epoch_pic_acc, time_elapsed // 60, time_elapsed % 60)) # Modify this line
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'Train Epoch GPU memory used: {info.used/1000000:.4f} MB') 
    return epoch_loss, epoch_acc, epoch_iou, epoch_pic_acc

def val_model(model, val_loader, metric, device):
    start_time = time.time()
    model.eval()

    nvmlInit()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    running_pic_acc =0.0

    for inputs, labels in tqdm(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

                
        with torch.no_grad():
            outputs = model(pixel_values=inputs,labels=labels)
            loss = outputs.loss
            predicted = outputs.logits.argmax(dim=1)
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
        
        metrics = metric.compute(num_labels=29,
                                ignore_index=0,
                                reduce_labels=False,
        )

        running_loss += loss.item() * inputs.size(0)
        running_iou += metrics["mean_iou"]
        running_acc += metrics["mean_accuracy"]
        running_pic_acc += calculate_pixel_accuracy(outputs.logits, labels, ignore_index=0)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_acc / len(val_loader.dataset)
    epoch_iou = running_iou / len(val_loader.dataset)
    epoch_pic_acc = running_pic_acc / len(val_loader.dataset)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Validation Loss: {:.4f} Acc: {:.4f} iou: {:.4f} pic_acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, epoch_iou, epoch_pic_acc, time_elapsed // 60, time_elapsed % 60)) # Modify this line
    return epoch_loss, epoch_acc, epoch_iou , epoch_pic_acc

def test_model(model, test_loader, metric, device):
    start_time = time.time()
    model.eval()

    nvmlInit()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    running_pic_acc =0.0

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)


        with torch.no_grad():
            outputs = model(pixel_values=inputs,labels=labels)
            loss = outputs.loss
            predicted = outputs.logits.argmax(dim=1)
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        metrics = metric.compute(num_labels=29,
                                ignore_index=0,
                                reduce_labels=False,
        )

        running_loss += loss.item() * inputs.size(0)
        running_iou += metrics["mean_iou"]
        running_acc += metrics["mean_accuracy"]
        running_pic_acc += calculate_pixel_accuracy(outputs.logits, labels, ignore_index=0)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_acc / len(test_loader.dataset)
    epoch_iou = running_iou / len(test_loader.dataset)
    epoch_pic_acc = running_pic_acc / len(test_loader.dataset)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Test Loss: {:.4f} Acc: {:.4f} iou: {:.4f} pic_acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(epoch_loss, epoch_acc, epoch_iou, epoch_pic_acc, time_elapsed // 60, time_elapsed % 60))
    return epoch_loss, epoch_acc, epoch_iou, epoch_pic_acc

if __name__ == '__main__':

    predicted_logits = torch.randn(16, 29, 224, 224)
    predicted_logits = predicted_logits.to('cuda')
    true_mask = torch.randint(0, 29, (16, 224, 224))
    true_mask = true_mask.to('cuda')
    print('logit shape: ', predicted_logits.size())
    print('mask shaep: ', true_mask.size())
    accuracy = calculate_pixel_accuracy(predicted_logits, true_mask, ignore_index=0)

    print(f"Pixel Accuracy (ignoring class 0): {accuracy * 100:.2f}%")







