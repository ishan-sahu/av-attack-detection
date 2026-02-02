from art.estimators.classification import BlackBoxClassifierNeuralNetwork
from safety.attacks.snal.steal_now_attack_later import SNAL
from ultralytics import YOLO
from art.estimators.object_detection import PyTorchYolo
import torch

import numpy as np
import time
import os
import pickle as pkl

class Attacker():
    def __init__(self, 
                 safety_config,
                 input_shape=(3, 480, 960)):
        self.safety_config = safety_config
        self.input_shape = input_shape  

        self.model = YOLO('yolov8m')
        self.py_model = PyTorchYolo(
            model=self.model,           
            input_shape=self.input_shape, 
            channels_first=True,
            is_yolov8=True
        )

        candidates_filename = self.safety_config.args['candidates_list_file']
        with open(candidates_filename, 'rb') as handle:
            self.candidates_list = pkl.load(handle)

        eps = float(self.safety_config.args['eps_n'])/255.0
        self.attacker = SNAL(
            estimator = self.py_model,
            eps = eps,
            max_iter = 10, # 5 for interfuser 
            num_grid = 5, # for height dimension as in custom code
            candidates = self.candidates_list,
            collector = collect_patches_from_images
        )
        
    def generate(self, input_array):
        x_adv = self.attacker.generate(input_array/255.0)*255.0
        return x_adv

    def destroy(self):
        del self.py_model
        del self.model
        return
    
def collect_patches_from_images(model: 'torch.nn.Module',
                                imgs: 'torch.Tensor'):
    """
    Collect patches and corresponding spatial information by the model from
    images.

    :param model: Object detection model
    :param imgs: Target images

    :return: Detected objects and corresponding spatial information
    """
    bs = imgs.shape[0]
    with torch.no_grad():
        #pred = model.model(imgs)
        pred = model.model.predict(imgs, verbose=False) # to turn off print statements
    y = []
    for obj in pred:
        y.append(obj.boxes.xyxy)

    candidates_patch = []
    candidates_position = []
    for i in range(bs):
        patch = []
        if y[i].shape[0] == 0:
            candidates_patch.append(patch)
            candidates_position.append(torch.zeros((0, 4), device=model.device))
            continue

        pos_matrix = y[i][:, :4].clone().int()
        pos_matrix[:, 0] = torch.clamp_min(pos_matrix[:, 0], 0)
        pos_matrix[:, 1] = torch.clamp_min(pos_matrix[:, 1], 0)
        pos_matrix[:, 2] = torch.clamp_max(pos_matrix[:, 2], imgs.shape[3])
        pos_matrix[:, 3] = torch.clamp_max(pos_matrix[:, 3], imgs.shape[2])
        for e in pos_matrix:
            p = imgs[i, :, e[1]:e[3], e[0]:e[2]]
            patch.append(p.to(model.device))
        
        candidates_patch.append(patch)
        candidates_position.append(pos_matrix)
    
    return candidates_patch, candidates_position
