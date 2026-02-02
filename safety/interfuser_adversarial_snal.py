import os
import json
from copy import deepcopy

import cv2
import carla
from PIL import Image
from PIL import ImageOps
from collections import deque

import torch
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from drivers.team_code_interfuser import interfuser_agent
from srunner.scenariomanager.timer import GameTime
from leaderboard.utils.route_manipulation import downsample_route
from safety_config import SafetyConfig

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

from safety.attacks.poltergeist.blur import cal_blur

import itertools
import pathlib

from safety.attacks.snal.attacker import Attacker

# this path to be used to save intermediate data useful for debugging
DEBUG_SAVE_PATH = os.environ.get('DEBUG_SAVE_PATH')

if not DEBUG_SAVE_PATH:
    DEBUG_SAVE_PATH = None
else:
    pathlib.Path(DEBUG_SAVE_PATH).mkdir(parents=True, exist_ok=True)

def get_entry_point():
    return 'AdversarialAgent'

class DisplayInterface(object):
    '''
    Display for simulation designed based on driving agents input-output
    '''
    def __init__(self): 
        pass

    def run_interface(self, input_data):
        pass

    def _quit(self):
        pygame.quit()

class AdversarialAgent(autonomous_agent.AutonomousAgent):
        
    def setup(self, path_to_conf_file):
        self.config_path = os.path.join(path_to_conf_file, 'safety_config.json')
        self.safety_config = SafetyConfig(self.config_path)
        self.num_skip_frames = self.safety_config.args['num_skip_frames']
        self.count_skip = 0

        # driving agent - da
        self.da = interfuser_agent.InterfuserAgent(self.safety_config.args['agent_config'])
        self.camera_ids = ['rgb', 'rgb_left', 'rgb_right']

        if self.safety_config.args['disp_interface'] == 1:
            self.disp_interface = DisplayInterface()
        else:
            self.disp_interface = None

        L_shape = (3, 600, 800)
        S_shape = (3, 300, 416) # padded shape - divisible by 32
        self.attackerL = Attacker(self.safety_config, input_shape=L_shape)
        self.attackerS = Attacker(self.safety_config, input_shape=S_shape)

        self.track = autonomous_agent.Track.SENSORS
        self.step = -1 # check in case of overflow, keep track of steps

    def sensors(self):
        return self.da.sensors()
    
    # AutonomousAgent overridden
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

        self.da.set_global_plan(global_plan_gps, global_plan_world_coord)

        # # for the blackbox agent - set the same plan as the driving agent
        # self.bb_agent.set_global_plan(global_plan_gps, global_plan_world_coord)

    
    #@torch.inference_mode() # Faster version of torch_no_grad
    def run_step(self, input_data, timestamp):
        self.step += 1

        # add adversarial modifications - black box and white box
        if self.count_skip == self.num_skip_frames:
            # modify input data

            # create input image array
            input_data = self.get_adversarial_data(input_data)

            self.count_skip = 0
        else:
            self.count_skip += 1

        #print(input_data.keys())

        # set driving agent wallclock
        if not self.da.wallclock_t0:
            self.da.wallclock_t0 = GameTime.get_wallclocktime()
        control = self.da.run_step(input_data, timestamp)
        if self.disp_interface is not None:
            self.disp_interface.run_interface(input_data)

        return control


    def get_adversarial_data(self, input_data):

        for rgb_cam in self.camera_ids:
            #print(input_data[rgb_cam][1][:, :, :3].shape)
            rgb_array = input_data[rgb_cam][1][:, :, :3]
            if rgb_cam == 'rgb':
                # 800 X 600
                rgb_array = self.pad_image(rgb_array, (0, 20, 0, 20))
            elif rgb_cam == 'rgb_left' or rgb_cam == 'rgb_right':
                # 400 X 300 - pad width both sides by 8 pixels 
                rgb_array = self.pad_image(rgb_array, (8, 10, 8, 10))
            else:
                print('Invalid rgb_cam!')
            input_array = np.expand_dims(rgb_array, axis=0)
            input_array = input_array.astype(np.float32)
            #print(input_array.shape)
            if rgb_cam == 'rgb':
                # 800 X 600
                adv_input = self.attackerL.generate(input_array.transpose((0, 3, 1, 2)))
            elif rgb_cam == 'rgb_left' or rgb_cam == 'rgb_right':
                # 400 X 300
                adv_input = self.attackerS.generate(input_array.transpose((0, 3, 1, 2)))
            else:
                print('Invalid rgb_cam!')
            adv_input = adv_input.transpose((0, 2, 3, 1))
            adv_input = adv_input[0]
            if rgb_cam == 'rgb':
                width = 800
                height = 600
                adv_input = self.crop_image(adv_input, (0, 20, width, height+20))
            elif rgb_cam == 'rgb_left' or rgb_cam == 'rgb_right':
                width = 400
                height = 300
                adv_input = self.crop_image(adv_input, (8, 10, width+8, height+10))
            else:
                print('Invalid rgb_cam!')
            input_data[rgb_cam][1][:, :, :3] = adv_input
        
        return input_data
    
    def pad_image(self, img_array, border):
        # for images with 400 X 300 shape - to make width (400) divisible 
        # by 32
        im = Image.fromarray(img_array)
        im_padded = ImageOps.expand(im,
                                    border=border,
                                    fill=0)
        return np.array(im_padded)
    
    def crop_image(self, img_array, box):
        # crop the adversarial image generated after padding to original
        # dimensions
        #width = 400
        #height = 300
        im = Image.fromarray(img_array.astype(np.uint8))
        im_cropped = im.crop(box=box)
        return np.array(im_cropped)

    def destroy(self):
        self.attackerL.destroy()
        self.attackerS.destroy()
        #self.bb_agent.destroy()
        self.da.destroy()
