"""
Created on Thr Apr 18 08:15:43 2019

@author: BATMAN/secg95
"""
import matplotlib
matplotlib.use('Agg')
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io, transform
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb
from torch.utils import data
from PIL import Image


class PlanetDataset(Dataset):
    """Amazing Planet dataset."""

    def __init__(self, csv_file, root_dir, extension, problem, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            extension : image extension of ther images to be loaded
            problem : either atmospheric or other.
        """
        #Tags de las imagenes en el directorio raiz, asi se pueden cargar facilmente directorios mas pequenos
        self.AmazonasImages = [x.split('.')[0] for x in os.listdir(root_dir)]
        #root directory for the data, such directory must have a train and validation directory
        self.root_dir = root_dir
        #image extensions
        self.extension = extension
        #csv files with image labels
        self.labels  = pd.read_csv(csv_file)
        #transforms to be made
        self.transform = transform
        # problem either atmospheric or other.
        self.problem = problem
        ## TODO assert of problem and rise exception

        #all posible labels
        self.atmospheric = ["clear","partly_cloudy","cloudy","haze"]
        self.common = ["agriculture","bare_ground","cultivation","habitation","primary","road","water"]
        self.rare = ["artisinal_mine","blooming","blow_down","conventional_mine", "selective_logging","slash_burn"]
        self.other_tags = self.common + self.rare

    def __len__(self):
        """
         lenm method, returns len of the data set.
        """
        return len(self.AmazonasImages)

    def __getitem__(self, index):
        #creating the path for laoding the images
        img_name = os.path.join(self.root_dir,
                                self.AmazonasImages[index]+
                                self.extension)
        #reading the image
        image = Image.open(img_name)
        #extracting the image label from the data frame
        Tag =  self.labels.loc[self.labels['image_name'] == self.AmazonasImages[index]]['tags'].iloc[0]
        #creation of the tags depending of the problem
        if (self.problem == 'atmospheric'):
            tags = 0
            #an integer depending of each on of the four atmospheric categories
            for j in range(len(self.atmospheric)):
                if self.atmospheric[j] in Tag:
                   tags = j
                   break
            #tag to be returned
            Tag = tags
        #TODO put an assert for the only two possible problems
        else:
            #temp variable to store the categorical not atmospheric tag.
            temp = 0
            #extracting the non atmospherical attributes
            Tag = [x for x in Tag.split(' ') if x in self.other_tags]
            #Join all the categories in one string
            Tag = ' '.join(Tag)
            #making a list of all possible non atmospheric tags TODO define this list as attribute of the class
            tags = list(set(self.labels['tags']))

            #tag extarction
            for i in range(len(tags)):
                if Tag in tags[i]:
                   temp = i
                   break
            #Tag signation to the variable that is going to be returned
            Tag = temp
        #apllying transforms to the image
        if self.transform:
            image  = self.transform(image)
        #return
        sample =[image,Tag]
        return sample
