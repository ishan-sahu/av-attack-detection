#%%
import os
import sys
sys.path.append('/home/ishan/dev/repos/github/av-attack-detection')
import pathlib

import cv2
from PIL import Image
import numpy as np

from safety.attacks.esia.attacker import Attacker
from safety.safety_config import SafetyConfig

#%%
filepath = '/home/ishan/dev/repos/github/av-attack-detection/experiments/scripts/configs/transfuser/esia/safety_config.json'
# set attack config
safety_config = SafetyConfig(filepath)
attacker = Attacker(safety_config=safety_config, input_shape=(480, 960, 3))

#%%
def get_adversarial_data(attacker, input_data):
    # camera_width = 960 # for transfuser agent
    # camera_height = 480 # for transfuser agent
    input_array = input_data[:, :, :3]

    adv_input_data = attacker.generate(input_array)

    return adv_input_data


#%%
#route = 'longest_weathers_0'
route = 'longest_weathers_19'
# input path for the clean recorded images
input_folder = f'/home/ishan/dev/repos/github/av-attack-detection/my_datagen/snt_2_ps/polter/{route}/original'

s = 1 # IMPORTANT: also change the safety config 
# s = 2
# s = 3 
output_folder = f'/home/ishan/dev/repos/github/av-attack-detection/my_datagen/snt_2_ps/esia/{route}/{s}'

#%%
def save_bgr_as_rgb(img, filename):
    im = Image.fromarray(img)
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    im.save(filename)

#%%
directory_path = pathlib.Path(input_folder)  # Replace with your directory

for file_path in directory_path.iterdir():
    if file_path.is_file():
        print(f"File: {file_path}")
        img = cv2.imread(file_path)
       
        adv_array = get_adversarial_data(attacker, img)
        
        filename = file_path.parts[-1]
        out_filename = filename.replace('original', 'esia')
        save_bgr_as_rgb(adv_array, os.path.join(output_folder, out_filename))
        #break

# %%
