#%%
import torch
import torch.nn as nn
import numpy as np
import detector_model as dm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import numpy as np

import sys
sys.path.append("/home/ishan/dev/repos/github/av-attack-detection")

import logging
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim.lr_scheduler import LRScheduler, StepLR
from tqdm import tqdm
from typing import Optional

from datetime import datetime
from time import time

# %%
from tools.detector_dataset.carla_adv_dataset import TemporalDataset
import detector_model as dm2
import torchinfo
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

#%%
train_route = 'longest_weathers_19'
train_dict_filename = f'{train_route}_combined_dataset_dict.json'
train_dict_path = os.path.join('/home/ishan/dev/repos/github/av-attack-detection/tools/detector_dataset/combined_dataset_jsons', train_dict_filename)

test_route = 'longest_weathers_0'
test_dict_filename = f'{test_route}_combined_dataset_dict.json'
test_dict_path = os.path.join('/home/ishan/dev/repos/github/av-attack-detection/tools/detector_dataset/combined_dataset_jsons', test_dict_filename)

base_path = '/home/ishan/dev/temp/dataset/'

# %%
preprocess_mean = np.array([0.485, 0.456, 0.406])
preprocess_std = np.array([0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=preprocess_mean, std=preprocess_std)
resize = transforms.Resize(size=(160, 320))

transformations = transforms.Compose([
    resize,
    normalize,
])

# load training data
train_dataset = TemporalDataset(train_dict_path,
                                 base_path,
                                 transform=transformations)

test_dataset = TemporalDataset(test_dict_path,
                                base_path,
                                transform=transformations)

# %%
num_workers = 4
shuffle = True
batch_size = 32

train_dataloader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
eval_dataloader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

#%%
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
model = dm2.CBDetector(num_classes=4,
                      dropout=0.2,
                      )
model.to(device)

#%%
print(model.feature_dim)

#%%
rows, labels = next(iter(train_dataloader))

# %%
#check_out = model(rows)

# %%
def train_model(model: nn.Module,
                device: str,
                train_dataloader,
                eval_dataloader,
                test_dataloader,
                epochs: int,
                optimizer: torch.optim.Optimizer,
                scheduler: Optional[LRScheduler],
                log_interval: int = 1,
                select_label: int = 0):
    model.train()

    criterion = CrossEntropyLoss()

    for epoch in tqdm(range(1, epochs + 1), 'Training', initial=1):
        model.train()
        total_loss = 0
        n_items = 0
        for x, labels in train_dataloader:
            model.zero_grad()
            labels = labels[1][select_label]
            labels = labels.to(device)
            x[0] = [e.to(device) for e in x[0]]
            x[1] = [e.to(device) for e in x[1]]
            predicted = model(x)
            loss = criterion(predicted, labels)
            total_loss += loss.item()
            #n_items += len(data)
            n_items += len(labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
            logging.info(f'scheduled lr={scheduler.get_last_lr()}')

        logging.info(f'Average running loss at epoch {epoch}/{epochs}: {total_loss / n_items}')

        if epoch % log_interval == 0:
            train_accuracy, _, _, _, _ = test_model(model, device, eval_dataloader, select_label=select_label)
            logging.info(f'Train accuracy/Train at epoch {epoch}: {train_accuracy}')

            test_accuracy, _, _, _, _ = test_model(model, device, test_dataloader, select_label=select_label)
            logging.info(f'Test accuracy/Train at epoch {epoch}: {test_accuracy}')
            path_model = f'cbdetector_{epoch}.pth'
            torch.save(model.state_dict(), path_model)


@torch.no_grad()
def test_model(model: nn.Module, device: str, test_dataloader,
               select_label=0):
    # during testing statistics using both class labels and fine labels
    
    model.eval()
    correct = 0
    total = 0
    predictions = None
    true_class_labels = None
    true_fine_labels = None
    total_time = 0
    for x, all_labels in test_dataloader:
        labels = all_labels[1][select_label] # here class accuracy is being computed
        labels = labels.to(device)
        x[0] = [e.to(device) for e in x[0]]
        x[1] = [e.to(device) for e in x[1]]
        t1 = time()
        prob = model(x)
        t2 = time()
        total_time += (t2-t1)
        y_hat = prob.argmax(dim=-1)
        correct += torch.sum(labels == y_hat).item()
        total += len(labels)
        if predictions is None:
            predictions = y_hat
            true_class_labels = all_labels[1][0]
            true_fine_labels = all_labels[1][1]
        else:
            predictions = torch.cat([predictions, y_hat], dim=0)
            true_class_labels = torch.cat([true_class_labels, all_labels[1][0]], dim=0)
            true_fine_labels = torch.cat([true_fine_labels, all_labels[1][1]], dim=0)

    test_accuracy = correct / total
    return test_accuracy, predictions, true_class_labels, true_fine_labels, total_time

def setup_logger():
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f'my_model_detect_attack_{now_str}.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/{filename}"),
            logging.StreamHandler()
        ]
    )
    return filename

#%%
filename = setup_logger()
logging.info(torchinfo.summary(model))
select_label = 1 # use fine labels - 4 class classification

##
lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)
epochs = 20

log_interval = 2

dir_saved_models = './take_models'
experiment_name = 'take_test_dm2_for_table'

os.makedirs(dir_saved_models, exist_ok=True)

train_model(model, device, 
                    train_dataloader, eval_dataloader, test_dataloader,
                    epochs=epochs, 
            optimizer=optimizer, scheduler=lr_scheduler, log_interval=log_interval, select_label=select_label)
path_model = f'{dir_saved_models}/{experiment_name}.pth'
logging.info(f'Model save file: {path_model}')
torch.save(model.state_dict(), path_model)

# %%
test_accuracy, predictions, true_class_labels, true_fine_labels, total_time = test_model(model, device, test_dataloader, select_label=select_label)

cm = confusion_matrix(true_fine_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy(), labels=[0, 1, 2, 3])
print(cm)

time_per_instance = total_time/len(true_fine_labels)
print(f'Time per instance: {time_per_instance}')

# %%
# load the saved model and test again
#dir_saved_models = './my_saved_models'
dir_saved_models = './take_models'
experiment_name = 'cbdetector_18'
device = 'cuda'
select_label = 1

path_model = f'{dir_saved_models}/{experiment_name}.pth'
model = dm2.CBDetector(num_classes=4,
                      dropout=0.2,
                      )
model.load_state_dict(torch.load(path_model, weights_only=True))
model.to(device)
model.eval()

# %%
