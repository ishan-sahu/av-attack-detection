import torch

import numpy as np
import time
import os
import cv2

class Attacker():
    def __init__(self, 
                 safety_config,
                 input_shape=(480, 960, 3),
                 input_type='bgr'):
        self.safety_config = safety_config
        self.input_shape = input_shape  

        self.severity = int(self.safety_config.args['severity'])
        self.min_size = 10
        self.input_type = input_type
        
    def generate(self, input_array):
        strips = self.get_strips()
        if self.input_type == 'bgr':
            new_array = self.simulate_esia_bgr(strips, input_array)
            demosaiced_image = cv2.cvtColor(new_array, cv2.COLOR_BAYER_BG2BGR)
        elif self.input_type == 'rgb':
            x_adv = self.simulate_esia_rgb(strips, input_array)
            demosaiced_image = cv2.cvtColor(new_array, cv2.COLOR_BAYER_BG2RGB)
        else:
            raise ValueError(f'Invalid colour image format {self.input_type}!')

        mask_for_original = demosaiced_image != 0
        x_adv = input_array.copy()
        x_adv[mask_for_original] = demosaiced_image[mask_for_original]
        return x_adv
    
    def get_strips(self):
        strips = {}
        if self.severity == 1: # mild
            num_strips = 3
        elif self.severity == 2: # moderate
            num_strips = 6
        elif self.severity == 3: # severe
            num_strips = 15
        else:
            print('Invalid severity! Defaulting to severe.')
            num_strips = 15
        
        max_width = int(self.input_shape[0]/num_strips)
        prev = 0
        for i in range(num_strips):
            start_i = np.random.randint(prev, prev+max_width-self.min_size-1)
            end_i = np.random.randint(start_i+self.min_size, prev+max_width)

            prev = prev+max_width
            strips[i] = list(range(start_i, end_i, 1))
        return strips
    
    def simulate_esia_rgb(self, strips, arr):
        # impacted_rows is a list of row indices
        inter_array = np.zeros(shape=self.input_shape[0:2], dtype=np.uint8)
        for strip, impacted_rows in strips.items():
            for k in range(0, len(impacted_rows), 1):
                i = impacted_rows[k]
                for j in range(0, self.input_shape[1], 2):
                    r = arr[i][j][0]
                    g1 = arr[i][j+1][1]
                    g2 = arr[i+1][j][1]
                    b = arr[i+1][j+1][2]
                    if i % 2 == 0: # i is even
                        r_i = g2
                        g1_i = b
                        inter_array[i][j] = r_i
                        inter_array[i][j+1] = g1_i
                    else: # i is odd
                        g2_i = r
                        b_i = g1
                        inter_array[i][j] = g2_i
                        inter_array[i][j+1] = b_i
                
        return inter_array
    
    def simulate_esia_bgr(self, strips, arr):
        # impacted_rows is a list of row indices
        inter_array = np.zeros(shape=self.input_shape[0:2], dtype=np.uint8)
        for strip, impacted_rows in strips.items():
            for k in range(0, len(impacted_rows), 1):
                i = impacted_rows[k]
                for j in range(0, self.input_shape[1], 2):
                    r = arr[i][j][2]    # for BGR images
                    g1 = arr[i][j+1][1]
                    g2 = arr[i+1][j][1]
                    b = arr[i+1][j+1][0]  # for BGR images
                    if i % 2 == 0: # i is even
                        r_i = g2
                        g1_i = b
                        inter_array[i][j] = r_i
                        inter_array[i][j+1] = g1_i
                    else: # i is odd
                        g2_i = r
                        b_i = g1
                        inter_array[i][j] = g2_i
                        inter_array[i][j+1] = b_i
                
        return inter_array

    def destroy(self):
        return
    
