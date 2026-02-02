#%%
import os
import sys
sys.path.append('/home/ishan/dev/repos/github/av-attack-detection')
import pathlib

import cv2
from PIL import Image
import numpy as np

from safety.attacks.snal.attacker import Attacker
from safety.safety_config import SafetyConfig

#%%
# set attack config
safety_config = SafetyConfig('/home/ishan/dev/repos/github/av-attack-detection/experiments/scripts/configs/transfuser/snal/safety_config.json')
attacker = Attacker(safety_config=safety_config)

#%%
def get_adversarial_data(attacker, input_data):
    # camera_width = 960 # for transfuser agent
    # camera_height = 480 # for transfuser agent
    # prepare input array of concatenated images
    input_array = input_data[:, :, :3]
    input_array = np.expand_dims(input_array, axis=0)
    input_array = input_array.astype(np.float32)

    # generate adversarial concatenated image
    # convert to channels first before attacker input (IMP: take care of batch size)
    adv_input_images = attacker.generate(input_array.transpose((0, 3, 1, 2)))
    # convert back to channels last before agent input
    adv_input_images = adv_input_images.transpose((0, 2, 3, 1))
    # separate concatenated adversarial image and update input data
    input_data[:, :, :3] = adv_input_images[0][:, :, :3]
    return input_data


#%%
#route = 'longest_weathers_0'
route = 'longest_weathers_19'
# input path for the clean recorded images
input_folder = f'/home/ishan/dev/repos/github/av-attack-detection/my_datagen/snt_2_ps/polter/{route}/original'

eps = '8'
#eps = '4'
output_folder = f'/home/ishan/dev/repos/github/av-attack-detection/my_datagen/snt_2_ps/snal/{route}/{eps}'

#%%
directory_path = pathlib.Path(input_folder)  # Replace with your directory

for file_path in directory_path.iterdir():
    if file_path.is_file():
        print(f"File: {file_path}")
        im = Image.open(file_path)
        #im.show()
        im_array = np.copy(np.asarray(im))
        adv_array = get_adversarial_data(attacker, im_array)
        adv_im = Image.fromarray(adv_array)

        filename = file_path.parts[-1]
        out_filename = filename.replace('original', 'snal')
        adv_im.save(os.path.join(output_folder, out_filename))
        #break

# %%
