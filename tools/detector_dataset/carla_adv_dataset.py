#%%
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode # try decode_image
decode_image = read_image
#from torchvision.io import decode_image

import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

import os
import json
import ast

#%%
class TemporalDataset(Dataset):
    def __init__(self,
                 dataset_filename,
                 base_path,
                 transform=None,
                 target_transform=None,
                 ):
        
        # load dataset json
        self.dataset_dict = json2dict(dataset_filename)

        self.base_path = base_path
        self.transform = transform
        self.target_transform = target_transform
   

    def load_data(self, row, timestep):
        d = row[timestep]
        image_left = self.read_transform(d['image_paths'][0])
        image_front = self.read_transform(d['image_paths'][1])
        image_right = self.read_transform(d['image_paths'][2])

        return (image_left, image_front, image_right), (d['label'], d['fine_label'])  

    def __len__(self):
        return len(self.dataset_dict.keys())

    def __getitem__(self, idx):
        row = self.dataset_dict[idx]

        # load t-1 data
        t_1_data, t_1_label = self.load_data(row, 't_1')

        # load t data
        t_data, t_label = self.load_data(row, 't')

        # return 
        return (t_1_data, t_data), (t_1_label, t_label)

    def read_transform(self, filename):
        path = os.path.join(self.base_path, filename)
        image = read_image(path, ImageReadMode.RGB) # try decode_image
        #image = decode_image(path, mode='RGB')
        if self.transform:
            image = image/255.0
            image = self.transform(image)
        return image[0:3, :, :]

#%%
def dict2json(filename, d):
    json_compatible_dict = {str(k): v for k, v in d.items()}

    def convert_4_json(x):
        if isinstance(x, tuple):
            return str(x)
        elif isinstance(x, np.int64):
            return int(x)

    with open(filename, 'w') as cj:
        json.dump(json_compatible_dict, 
                cj, 
                indent=4, 
                #sort_keys=True, 
                default=convert_4_json)

def json2dict(filename):
    with open(filename, 'r') as f:
        loaded_json_data = json.load(f)
        reconstructed_dict = {ast.literal_eval(k): v for k, v in loaded_json_data.items()}

        return reconstructed_dict

#%%
from torchvision.models import resnet50
import torch.nn as nn

class KPCAImageDataset(Dataset):
    def __init__(self,
                 dataset_filename,
                 base_path,
                 transform=None,
                 target_transform=None,
                 filter_label=None,
                 filter_fine_label=None,
                 ):
        
        # load dataset json
        self.dataset_dict = json2dict(dataset_filename)

        # select only desired classes if applicable
        if filter_label is not None:
            if type(filter_label) is list:
                self.dataset_dict = self.filter_label(filter_label)
            else:
                raise TypeError(f'filter_label should be of type list. Received type: {type(filter_label)}')
        if filter_fine_label is not None:
            if type(filter_fine_label) is list:
                self.dataset_dict = self.filter_fine_label(filter_fine_label)
            else:
                raise TypeError(f'filter_fine_label should be of type list. Received type: {type(filter_fine_label)}')

        self.base_path = base_path
        self.transform = transform
        self.target_transform = target_transform
        # self.model = resnet50(weights='IMAGENET1K_V1')
        # self.model.fc = nn.Sequential()
        # self.model.to('cuda')
        # self.model.eval()
        

    def load_data(self, row, timestep):
        d = row[timestep]
        image_left = self.read_transform(d['image_paths'][0])
        image_front = self.read_transform(d['image_paths'][1])
        image_right = self.read_transform(d['image_paths'][2])

        image = torch.cat((image_left, image_front, image_right), dim=2)

        return image, (d['label'], d['fine_label'])  

    def __len__(self):
        return len(self.dataset_dict.keys())

    def __getitem__(self, idx):
        row = self.dataset_dict[idx]

        # load t data
        t_data, t_label = self.load_data(row, 't')

        # return 
        return t_data, t_label

    def read_transform(self, filename):
        path = os.path.join(self.base_path, filename)
        image = read_image(path,  ImageReadMode.RGB) # try decode_image
        #image = decode_image(path, mode='RGB')
        if self.transform:
             image = image/255.0
             image = self.transform(image)
        return image[0:3, :, :]
    
    def filter_label(self, filter_label):
        temp_dataset_dict = {}

        j = 0
        for k, v in self.dataset_dict.items():
            if v['t']['label'] in filter_label:
                temp_dataset_dict[j] = v
                j += 1
        
        return temp_dataset_dict
    
    def filter_fine_label(self, filter_fine_label):
        temp_dataset_dict = {}

        j = 0
        for k, v in self.dataset_dict.items():
            if v['t']['fine_label'] in filter_fine_label:
                temp_dataset_dict[j] = v
                j += 1
        
        return temp_dataset_dict
   
class OnlyImageDataset(Dataset):
    def __init__(self,
                 dataset_filename,
                 base_path,
                 transform=None,
                 target_transform=None,
                 filter_label=None,
                 filter_fine_label=None
                 ):
        
        # load dataset json
        self.dataset_dict = json2dict(dataset_filename)

        # select only desired classes if applicable
        if filter_label is not None:
            if type(filter_label) is list:
                self.dataset_dict = self.filter_label(filter_label)
            else:
                raise TypeError(f'filter_label should be of type list. Received type: {type(filter_label)}')
        if filter_fine_label is not None:
            if type(filter_fine_label) is list:
                self.dataset_dict = self.filter_fine_label(filter_fine_label)
            else:
                raise TypeError(f'filter_fine_label should be of type list. Received type: {type(filter_fine_label)}')

        self.base_path = base_path
        self.transform = transform
        self.target_transform = target_transform
        

    def load_data(self, row, timestep):
        d = row[timestep]
        image_left = self.read_transform(d['image_paths'][0])
        image_front = self.read_transform(d['image_paths'][1])
        image_right = self.read_transform(d['image_paths'][2])

        return (image_left, image_front, image_right), (d['label'], d['fine_label'])  

    def __len__(self):
        return len(self.dataset_dict.keys())

    def __getitem__(self, idx):
        row = self.dataset_dict[idx]

        # load t data
        t_data, t_label = self.load_data(row, 't')

        # return 
        return t_data, t_label

    def read_transform(self, filename):
        path = os.path.join(self.base_path, filename)
        image = read_image(path,  ImageReadMode.RGB) # try decode_image
        #image = decode_image(path, mode='RGB')
        if self.transform:
            image = self.transform(image)
        return image[0:3, :, :]
    
    def filter_label(self, filter_label):
        temp_dataset_dict = {}

        j = 0
        for k, v in self.dataset_dict.items():
            if v['t']['label'] in filter_label:
                temp_dataset_dict[j] = v
                j += 1
        
        return temp_dataset_dict
    
    def filter_fine_label(self, filter_fine_label):
        temp_dataset_dict = {}

        j = 0
        for k, v in self.dataset_dict.items():
            if v['t']['fine_label'] in filter_fine_label:
                temp_dataset_dict[j] = v
                j += 1
        
        return temp_dataset_dict