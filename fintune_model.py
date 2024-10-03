import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from functools import partial
import time
from tqdm import tqdm

import argparse
import matplotlib.pyplot as plt
import matplotlib

import evaluate

import myutils
import Mymodel




def main(args):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    print("cuda available:", use_cuda)
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get data loader
    train_loader, val_loader, test_loader = myutils.getdata(batch_size=args.batch_size, data_path=args.data_dir)

    model = Mymodel.load_model(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    matric = evaluate.load("mean_iou")

    # Train the model
    best_acc = 0.0
    train_loss = []
    val_loss = []
    for epoch in range(args.num_epochs):

        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        train_epoch_loss, train_epoch_acc, train_epoch_iou, train_pic_acc = myutils.train_model(model, train_loader, optimizer, matric, device)
        train_loss.append(train_epoch_loss)
        val_epoch_loss, val_epoch_acc, val_epoch_iou, val_pic_acc = myutils.val_model(model, val_loader, matric, device)
        val_loss.append(val_epoch_loss)
        if val_pic_acc > best_acc:
            best_acc = val_pic_acc
            torch.save(model.state_dict(), f'{args.log_dir}/{args.num_epochs}_best_model.pth') 

    try:
        # Plot the loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, args.num_epochs + 1), train_loss, marker='o', linestyle='-', color='b')
        plt.plot(range(1, args.num_epochs + 1), val_loss, marker='o', linestyle='-', color='g')
        plt.title(f'Training and validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.grid(True)
        plt.savefig(f"{args.log_dir}/{args.num_epochs}_vitb16_loss.png")
        # plt.show()
        plt.close()

    except Exception as e:
        print(f"Error plotting acc and loss E: {e}")

    # run test data
    
    #best_model = myutils.myDinoV2(os.path.join(args.log_dir, 'best_model.pth'))
    test_loss, test_acc, test_iou, test_pic_acc = myutils.test_model(model, test_loader, matric, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune ViT small on cityscapes')
    parser.add_argument('--type', default='aclip_rand',
                        help='model type')
    parser.add_argument('--batch_size', '-b', default=64, type=int, metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--log_dir', default='./myoutput', type=str, metavar='PATH',
                        help='path to directory where to log')
    parser.add_argument('--data_dir', default="./cityscapes", type=str,
                        help='path to the dataset')
    parser.add_argument('--learning_rate',
                        type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--weight_path', 
                        default='./pretrain/dinov2_vits14_pretrain.pth',
                        type=str,
                        help='path to the model weight')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    main(args)


