import os
import json
from copy import deepcopy

import cv2
import carla
from PIL import Image
from collections import deque

import torch
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from drivers.team_code_transfuser import submission_agent
from srunner.scenariomanager.timer import GameTime
from leaderboard.utils.route_manipulation import downsample_route

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

from safety.attacks.poltergeist.blur import cal_blur

import itertools
import pathlib

import sys

# this path to be used to save intermediate data useful for debugging
DEBUG_SAVE_PATH = os.environ.get('DEBUG_SAVE_PATH')

if not DEBUG_SAVE_PATH:
    DEBUG_SAVE_PATH = None
else:
    pathlib.Path(DEBUG_SAVE_PATH).mkdir(parents=True, exist_ok=True)

def get_entry_point():
    return 'AdversarialAgent'

from safety_config import SafetyConfig

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

        self.datagen_path = self.safety_config.args.get('datagen_path',
                                                        None)

        # driving agent - da
        self.da = submission_agent.HybridAgent(self.safety_config.args['agent_config'])
        if self.safety_config.args['disp_interface'] == 1:
            self.disp_interface = DisplayInterface()
        else:
            self.disp_interface = None

        self.track = autonomous_agent.Track.SENSORS
        self.step = -1 # check in case of overflow, keep track of steps

        # to save speed
        speed_filename = os.path.join(self.datagen_path, 'speed_records.txt')
        self.speed_file = open(speed_filename, 'w')

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

    
    #@torch.inference_mode() # Faster version of torch_no_grad
    def run_step(self, input_data, timestamp):
        self.step += 1

        fps = 20
        if self.step % fps == 0 or (self.step-1) % fps == 0:
            for pos in ['left', 'front', 'right']:
                rgb_cam = f'rgb_{pos}'
                img_filename = os.path.join(self.datagen_path,
                                            'original',
                                            f'{self.step}_{rgb_cam}_original.png')
                self.save_bgra_as_rgba(input_data[rgb_cam][1],
                                    img_filename
                                    )
            
            # write speed as measured by speedometer to file
            speed = input_data['speed'][1]['speed']
            self.speed_file.write(f'{self.step},{speed}\n')

        # add adversarial modifications - black box and white box
        if self.count_skip == self.num_skip_frames:
            # modify input data

            # use only original input_data to drive agent
            if self.step % fps == 0 or (self.step-1) % fps == 0:
                adv_input_data = self.prepare_adversarial_data(input_data)

                for pos in ['left', 'front', 'right']:
                    rgb_cam = f'rgb_{pos}'
                    img_filename = os.path.join(self.datagen_path,
                                                'polter',
                                                f'{self.step}_{rgb_cam}_polter.png')
                    self.save_bgra_as_rgba(adv_input_data[rgb_cam][1], 
                                        img_filename
                                        )

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

        # if self.step >= fps*500:
        #     sys.exit()


        return control

    def prepare_adversarial_data(self, input_data):
        for pos in ['left', 'front', 'right']:
            rgb_cam = 'rgb_' + pos
            input_data[rgb_cam+'_original'] = deepcopy(input_data[rgb_cam])
            input_data[rgb_cam][1][:, :, :3] = cal_blur(input_data[rgb_cam][1][:, :, :3],
                                                        20,
                                                        0,
                                                        0)

        return input_data

    def destroy(self):
        self.speed_file.close()
        self.da.destroy()

    def save_bgra_as_rgba(self, img, filename):
        im = Image.fromarray(img)
        b, g, r, a = im.split()
        im = Image.merge("RGBA", (r, g, b, a))
        im.save(filename)
        