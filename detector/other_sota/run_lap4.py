#%% 
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import pandas as pd
import sklearn.decomposition
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import os
import gc
import pickle
import cv2 as cv

#%%
import sys
sys.path.append("/home/ishan/dev/repos/github/av-attack-detection")

from tools.detector_dataset.carla_adv_dataset import OnlyImageDataset

import detector.pycm as pycm

#%%
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    #image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return cv.Laplacian(image, cv.CV_16S).var()

#%%
# %%
train_route = 'longest_weathers_19'
train_dict_filename = f'{train_route}_combined_dataset_dict.json'
train_dict_path = os.path.join('/home/ishan/dev/repos/github/av-attack-detection/tools/test_detector/combined_dataset_jsons', train_dict_filename)

test_route = 'longest_weathers_0'
test_dict_filename = f'{test_route}_combined_dataset_dict.json'
test_dict_path = os.path.join('/home/ishan/dev/repos/github/av-attack-detection/tools/test_detector/combined_dataset_jsons', test_dict_filename)

base_path = '/home/ishan/dev/datasets/snt_2_ps/'

#%%
resize = transforms.Resize(size=(160, 320))

transformations = transforms.Compose([
    resize
])

train_dataset = OnlyImageDataset(train_dict_path,
                                 base_path,
                                 transform=transformations)

test_dataset = OnlyImageDataset(test_dict_path,
                                base_path,
                                transform=transformations)

#%%
# %%
num_workers = 4
shuffle = True
batch_size = 32

train_dataloader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

#%%
def pt_2_cv_gray(tensor):
    numpy_image = tensor.numpy()

    # Convert the numpy array to a cv2 image
    cv_image = np.transpose(numpy_image, (1, 2, 0))
    cv_image = cv.cvtColor(cv_image, cv.COLOR_RGB2GRAY)

    return cv_image

def get_lap4_batch(images, labels):
    labels = labels[1]
    
    batch_vols = []
    for i in range(batch_size):
        try:
            vol_l = variance_of_laplacian(pt_2_cv_gray(images[0][i]))
            vol_f = variance_of_laplacian(pt_2_cv_gray(images[1][i]))
            vol_r = variance_of_laplacian(pt_2_cv_gray(images[2][i]))
            vol = (vol_l, vol_f, vol_r)
            
            batch_vols.append(vol)

        except Exception as e:
            #print(i)
            #print(batch_size)
            #print(images[1][0][i].shape)
            print(e)
    return np.array(batch_vols), labels.detach().cpu().numpy()

def generate_lap4(dataloader):
    batch_i = 0

    for images, labels in dataloader:
        batch_vols, batch_fine_labels = get_lap4_batch(images, labels)
        if batch_i == 0:
            all_vols = batch_vols
            all_fine_labels = batch_fine_labels
        else:
            all_vols = np.concatenate((all_vols, batch_vols),
                                      axis=0)
            all_fine_labels = np.concatenate((all_fine_labels, batch_fine_labels),
                                             axis=0)

        batch_i += 1

    return all_vols, all_fine_labels

# %%
train_dv, train_dv_fl = generate_lap4(train_dataloader)
test_dv, test_dv_fl = generate_lap4(test_dataloader)

#%%
def check_model(train_pdv, train_pdv_cl, test_pdv, test_pdv_cl):
    model = LogisticRegression()
    model.fit(train_pdv, train_pdv_cl)

    train_acc = model.score(train_pdv, train_pdv_cl)
    print(f'Train accuracy: {train_acc}')

    test_acc = model.score(test_pdv, test_pdv_cl)
    print(f'Test accuracy: {test_acc}')

    pred_test_prob = model.predict_proba(test_pdv)
    test_predictions = model.predict(test_pdv)

    cm = confusion_matrix(test_pdv_cl, test_predictions, labels=[0, 1, 2, 3])
    print('Confusion matrix:')
    print(cm)

    cm2 = pycm.ConfusionMatrix(actual_vector=test_pdv_cl,
                        predict_vector=test_predictions)

    cm2.print_matrix()
    cm2.stat(summary=True)

    # print train and test data distribution
    print('Train distribution:')
    for i in [0, 1, 2]:
        print(f'{i}: {np.sum(train_pdv_cl == i)}')

    print('Test distribution:')
    for i in [0, 1, 2]:
        print(f'{i}: {np.sum(test_pdv_cl == i)}')

#%%
check_model(train_dv, train_dv_fl, test_dv, test_dv_fl)
# %%
