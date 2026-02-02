#%%
import carla_adv_dataset as cad

import os
import shutil
import itertools
import pandas as pd
import json
import ast
import numpy as np
import random
import matplotlib.pyplot as plt

#%%
base_path = '/home/ishan/dev/temp/dataset'
variations = ['original', 'polter', 'snal', 'esia'] # different class types

# different attack hyperparameters
sub_variations = {
    'original': ['original'],
    'polter': ['polter'],
    'snal': ['4', '8'],
    'esia': ['1', '2', '3']
}

all_combinations = list(itertools.product(variations, variations))

# longest_weathers_0 is for testing and longest_weathers_19 is from training
routes = ['longest_weathers_0', 'longest_weathers_19'] 

#%%
# run the same steps once for each route
route = routes[0]
speed_file_path = os.path.join(base_path, f'{route}_speed_records.txt')

# %%
# read the speed records file
df_speed = pd.read_csv(speed_file_path, header=None, names=['frame_id', 'speed'])

# %%
# prepare dataset element indices with speed (speed is common for all)
indices = {}
j = 0 
for i in range(0, len(df_speed['frame_id']), 2):
    indices[j] = {
        't_1': {'id': df_speed['frame_id'][i], 'speed': df_speed['speed'][i]},
        't': {'id': df_speed['frame_id'][i+1], 'speed': df_speed['speed'][i+1]}
    }
    j += 1

# %%
# write to json
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
def show_pt_image(tensor):
    plt.imshow(tensor.permute(1, 2, 0))
    plt.show

#%%
# create balanced combined dataset of all different hyperparameters
# create combined dataset
num_hyperparameters = 2 # NOTE: 2 is for snal, 3 is for esia
to_combine = ['snal', 'esia']

# run once each for snal and esia
combine_type = to_combine[0]

num_instances = int(len(df_speed['frame_id'])/2)
to_select = int(num_instances/num_hyperparameters)

elements = list(range(num_instances))
selected = {}
for i in range(num_hyperparameters-1):
    selected[i] = random.sample(elements, to_select)
    elements = list(set(elements) - set(selected[i]))

selected[i+1] = elements

# check for disjoint selection
for key1, item1 in selected.items():
    for key2, item2 in selected.items():
        print(f'{key1} - {key2} = {len(set(item1) - set(item2))}')

dict2json(f'{combine_type}_{route}_selected.json', selected)

#%%
# copy selected frames from different hyperparameters in a common folder
# do this once each for snal and esia
combined_path = os.path.join(base_path, combine_type, route, 'combined')
base_combine_type_path = os.path.join(base_path, combine_type, route)
os.makedirs(combined_path, exist_ok=True)
j = 0 
for i in range(0, len(df_speed['frame_id']), 2):
    for k, l in selected.items():
        if j in l:
            sv = sub_variations[combine_type][k]
            for loc in ['left', 'front', 'right']:
                src_path = os.path.join(base_combine_type_path, sv, f'{df_speed["frame_id"][i]}_rgb_{loc}_{combine_type}.png')
                shutil.copy(src_path, combined_path)
                src_path = os.path.join(base_combine_type_path, sv, f'{df_speed["frame_id"][i+1]}_rgb_{loc}_{combine_type}.png')
                shutil.copy(src_path, combined_path)
    j += 1

#%%
# By now the files should be arranged as described in the readme file.

#%%
# now create dataset from already selected data
clean = 'original'
attacks = ['polter', 'snal', 'esia']

all_combinations = [] # combinations of t-1 and t set of image frames
all_combinations.append((clean, clean))
for a in attacks:
    all_combinations.append((clean, a))
    all_combinations.append((a, clean))
    all_combinations.append((a, a))

# %%
classwise_relative_path = {
    'original': 'original',
    'polter': 'polter',
    'snal': 'combined',
    'esia': 'combined'
}

class_label_dict = {
    'original': 0,
    'polter': 1,
    'snal': 1,
    'esia': 1   
}

fine_label_dict = {
    'original': 0,
    'polter': 1,
    'snal': 2,
    'esia': 3
}

#%%
# create json files where combinations are classwise separated - to be combined together later
classwise_data_indices = {}
for c in all_combinations:
    type_t_1 = c[0]
    type_t = c[1]
    base_path_t_1 = os.path.join(type_t_1, route, classwise_relative_path[type_t_1])
    base_path_t = os.path.join(type_t, route, classwise_relative_path[type_t])

    t_dict = {}
    for key, value in indices.items():
        t_1_id = value['t_1']['id']
        t_id = value['t']['id']
        t_dict[key] = {
            't_1': {
                'id': t_1_id,
                'image_paths': [
                    f'{base_path_t_1}/{t_1_id}_rgb_left_{type_t_1}.png',
                    f'{base_path_t_1}/{t_1_id}_rgb_front_{type_t_1}.png',
                    f'{base_path_t_1}/{t_1_id}_rgb_right_{type_t_1}.png'
                ], # left, front, and right in that order
                'speed': value['t_1']['speed'],
                'label': class_label_dict[c[0]],
                'fine_label': fine_label_dict[c[0]]
            },
            't': {
                'id': t_id,
                'image_paths': [
                    f'{base_path_t}/{t_id}_rgb_left_{type_t}.png',
                    f'{base_path_t}/{t_id}_rgb_front_{type_t}.png',
                    f'{base_path_t}/{t_id}_rgb_right_{type_t}.png'
                ], # left, front, and right in that order
                'speed': value['t']['speed'],
                'label': class_label_dict[c[1]],
                'fine_label': fine_label_dict[c[1]]
            }
        }
    classwise_data_indices[c] = t_dict

dataset_dict_filename = f'{route}_classwise_dict.json'
dict2json(dataset_dict_filename, classwise_data_indices)

# %%
# combine json for dataset - this json file will be used for loading the dataset
dataset_dict = {}
i = 0
for key, item in classwise_data_indices.items():
    for index, row in item.items():
        dataset_dict[i] = row
        i += 1 

dataset_dict_filename = f'{route}_combined_dataset_dict.json'
dict2json(dataset_dict_filename, dataset_dict)

# %%
# test the created json file by loading the dataset
from carla_adv_dataset import TemporalDataset
dataset_dict_filename = f'{route}_combined_dataset_dict.json'
temporal_dataset = TemporalDataset(dataset_filename=os.path.join('combined_dataset_jsons', dataset_dict_filename),
                                   base_path=base_path)

# %%
import matplotlib.pyplot as plt
def show_pt_image(tensor):
    plt.imshow(tensor.permute(1, 2, 0))
    plt.show

#%%
from torch.utils.data import DataLoader
temporal_dataloader = DataLoader(temporal_dataset,
                                 batch_size=8,
                                 shuffle=True)

#%%
rows, labels = next(iter(temporal_dataloader))

# %%
# left image
show_pt_image(rows[0][0][0])

# %%
# centre image
show_pt_image(rows[0][1][0])

# %%
# right image
show_pt_image(rows[0][2][0])

# %%
