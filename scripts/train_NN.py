from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from create_planet_dataset import PlanetDataset
import torchvision
from torchvision import datasets, models, transforms
from models import initialize_model
from train_function import train_model
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import copy
import argparse
## import tqdm
## import visdom

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

parser = argparse.ArgumentParser('DO SOMETHING!')

parser.add_argument('--model', type=str, required=True, choices=['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'])
parser.add_argument('--epoch', type=int, default=15)
parser.add_argument('--batch', type=int, default=24)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
args = parser.parse_args()

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = args.model
# Label problem, atmospheric or others
problem = 'other'
# Batch size for training (change depending on how much memory you have)
batch_size = args.batch
# Number of epochs to train for
num_epochs = args.epoch
# Number of classes
num_classes = 0
# Name of file
name_file = args.name
# Resume checkpoint
model_resume = args.resume
#labels
if (problem == 'atmospheric'):
    num_classes = 4
else:
    labels = pd.read_csv('../data/train_v2.csv')
    num_classes = len(list(set(labels['tags'])))

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = True

#selecting the model
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)
#TODO split data in code not in directories

# Create training and validation datasets
image_datasets = {x: PlanetDataset(csv_file='../data/train_v2.csv',
                                    root_dir='../data/fast_' + x,
                                    extension='.jpg',problem=problem , transform=transforms.Compose([
                                              transforms.Resize(input_size),transforms.ToTensor()])) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=1) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
#TODO modify the loss function.
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft = train_model(name_file, model_resume, model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, device = device, is_inception=(model_name=="inception"))
