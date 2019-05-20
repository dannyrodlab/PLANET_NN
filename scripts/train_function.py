from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
## import pdb
## import tqdm
## import visdom

def save_checkpoint(state, filename):
    print("=> saving checkpoint '{}'".format(filename))
    filename_checkpoint = 'checkpoints/'+ filename + '_checkpoint.pth.tar'
    if os.path.isfile(filename_checkpoint):
        filename_checkpoint = 'checkpoints/'+ filename + '_new_checkpoint.pth.tar'
        ## torch.save(state, filename_checkpoint)
        print('Come on!')
    else:
        ## torch.save(state, filename_checkpoint)
        print('Come on two!')

def train_model(name, resume, model, dataloaders, criterion, optimizer, device, num_epochs, is_inception=False):
    """
       Train_model function loads the dataloa der to the choosen model, run train and validation steps
       model: model choosen
       dataloaders: a dictionary of data loaders one for the training epochs and on efor the validation ones
       criterion  loss function
       optimizer: derivates for backpropagation
       device: GPU or CPU
    """
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            ## start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    since = time.time()

    history_acc = []
    history_loss = []
    history_val_acc = []
    history_val_loss = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
            ## batch_idx, (inputs, labels) in tqdm.tqdm(enumerate(dataloaders[phase]),total=len(dataloaders[phase]), desc="Epoch: {}".format(epoch)):

                inputs = inputs[:,0:3,:,:]
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.5f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                history_acc.append(epoch_acc)
                history_loss.append(epoch_loss)
            if phase == 'val':
                history_val_acc.append(epoch_acc)
                history_val_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict()}, 
                                name)

    ## print()
    ## name_file = 'vgg' + '.txt' 
    ## datetime.datetime.now().strftime("%y%m%d%H%M") +
    time_elapsed = time.time() - since
    np.savetxt('./results/' + name + '.txt', (history_acc, history_loss, history_val_acc, history_val_loss), delimiter=',')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    ## model.load_state_dict(best_model_wts)
    return model
