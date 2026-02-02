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

#%%
import sys
sys.path.append("/home/ishan/dev/repos/github/av-attack-detection")

from tools.detector_dataset.carla_adv_dataset import KPCAImageDataset

import detector.pycm as pycm

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
kpca_train_dataset = KPCAImageDataset(train_dict_path,
                                 base_path,
                                 filter_label=[0],
                                 transform=transformations) # only clean images

classifier_train_dataset = KPCAImageDataset(train_dict_path,
                                 base_path,
                                 transform=transformations)

test_dataset = KPCAImageDataset(test_dict_path,
                                base_path,
                                transform=transformations)

# %%
num_workers = 4
shuffle = True
batch_size = 32

argsM = 4096

kpca_train_dataloader = DataLoader(kpca_train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
classifier_train_dataloader = DataLoader(classifier_train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)

#%%
from torchvision.models import resnet50, resnet18
import torch.nn as nn

device = 'cuda' 
model = resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Sequential()
model.to(device)
model.eval()

def get_feat_vecs(model, x, device='cuda'):
    out = model(x.to(device))
    return out.detach().to('cpu')


# %%
# Define Kernel PCA transformations for both training and testing
kpca_algo = 'CoP'
#kpca_algo = 'CoRP'
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

#%%
i = 0
for imgs, _ in kpca_train_dataloader:
    v = get_feat_vecs(model, imgs)
    if i == 0:
        i+=1
        feat_vecs = v
    else:
        feat_vecs = torch.cat((feat_vecs, v), dim=0)

# %%
# train pca
# Q: first prepros or first reshape
# imgs, _ = next(iter(kpca_train_dataloader))
# feat_vecs = get_feat_vecs(model, imgs)
ftrain = prepos_feat(feat_vecs) # CoP

print(f'Ftrain shape: {ftrain.shape}')

#%%
with open(f'kpca_Ftrain.pkl', 'wb') as file:
    pickle.dump(ftrain, file)

#%%
# load ftrain
with open(f'kpca_Ftrain.pkl', 'rb') as f:
    ftrain = pickle.load(f)

#%%
# CoRP
gamma = 3 # same for TESTING
M = argsM # same for TESTING # 4096
if kpca_algo == 'CoRP':
    m = ftrain.shape[1]
    # gamma, M = args.gamma, args.M
    # same for TESTING
    w = np.sqrt(2*gamma)*np.random.normal(size=(M,m))   # generate M i.i.d. samples from p(w)
    u = 2 * np.pi * np.random.rand(M)
    ftrain = np.sqrt(2/M)*np.cos((ftrain.dot(w.T)+u[np.newaxis,:]))
    # ftest = np.sqrt(2/M)*np.cos((ftest.dot(w.T)+u[np.newaxis,:]))
    # for ood_dataset, food in food_all.items():
    #     food_all[ood_dataset] = np.sqrt(2/M)*np.cos((food.dot(w.T)+u[np.newaxis,:]))
    print()
    print("Method: {}".format(kpca_algo))
    print("gamma = %f, M = %d"%(gamma, M))
else:
    print()
    print("Method: {}".format(kpca_algo))

# %%
# -------- centralize the mapped features
mu = ftrain.mean(axis=0) # same for TESTING
ftrain = ftrain - mu

#%%
#%%
# # -------- linear PCA
# print()
# print("Running linear PCA...")
exp_var_ratio = 0.85


# n_components = kpca_batch_size
# pca = sklearn.decomposition.PCA(n_components=n_components)
# pca.fit(ftrain)
# K = ftrain.T.dot(ftrain)
# u_full, s, _ = np.linalg.svd(K)
# # ---- the reduction dimension q is
# # ---- selected according to the explained variance ratio
# s = pca.explained_variance_ratio_
# q, s_accuml = -1, np.zeros(ftrain.shape[1])
# for i in range(ftrain.shape[1]):
#     s_accuml[i] = sum(s[:i]) / sum(s)
#     if i > 0 and q < 0:
#         if s_accuml[i-1] < exp_var_ratio and s_accuml[i] >= exp_var_ratio:
#             q = i
# print("Linear PCA finished.")
# print("explained variance ratio = %f"%exp_var_ratio)
# print("reduction dimension    q = %d"%q)
# print("s_accuml at q-1 = %f"%s_accuml[q-1])
# print("s_accuml at q   = %f"%s_accuml[q])
# print("s_accuml at q+1 = %f"%s_accuml[q+1])
# -------- linear PCA
print()
print("Running linear PCA...")
K = ftrain.T.dot(ftrain)
u_full, s, _ = np.linalg.svd(K)
# ---- the reduction dimension q is
# ---- selected according to the explained variance ratio
q, s_accuml = -1, np.zeros(ftrain.shape[1])
for i in range(ftrain.shape[1]):
    s_accuml[i] = sum(s[:i]) / sum(s)
    if i > 0 and q < 0:
        if s_accuml[i-1] < exp_var_ratio and s_accuml[i] >= exp_var_ratio:
            q = i
print("Linear PCA finished.")
print("explained variance ratio = %f"%exp_var_ratio)
print("reduction dimension    q = %d"%q)
print("s_accuml at q-1 = %f"%s_accuml[q-1])
print("s_accuml at q   = %f"%s_accuml[q])
print("s_accuml at q+1 = %f"%s_accuml[q+1])

# -------- reconstruction error for OoD detection
u_q = u_full[:,:q]

with open(f'kpca_linear_{kpca_algo}_u_q.pkl', 'wb') as file:
    pickle.dump(u_q, file)

#%%
# save preprocessing parameters
with open(f'kpca_{kpca_algo}_mu.pkl', 'wb') as file:
    pickle.dump(mu, file)

if kpca_algo == 'CoRP':
    with open(f'kpca_{kpca_algo}_w.pkl', 'wb') as file:
        pickle.dump(w, file)
    
    with open(f'kpca_{kpca_algo}_u.pkl', 'wb') as file:
        pickle.dump(u, file)

    with open(f'kpca_{kpca_algo}_gamma.pkl', 'wb') as file:
        pickle.dump(gamma, file)
    
    with open(f'kpca_{kpca_algo}_M.pkl', 'wb') as file:
        pickle.dump(M, file)

#%%
# generate reconstruction error and train logistic regression
del(feat_vecs)
del(kpca_train_dataloader)
del(kpca_train_dataset)
gc.collect()

#%%
# use already calculated mu, w, u
def generate_reconstruction_errors(u_q, dataloader, mu, model, 
                                   kpca_algo='CoP', 
                                   w=None, u=None, M=None):
    batch_i = 0

    for batch_imgs, batch_labels in dataloader:
        # print(batch_i)
        test_imgs = get_feat_vecs(model, batch_imgs)
        test_imgs = prepos_feat(test_imgs)

        if kpca_algo == 'CoRP':
            test_imgs = np.sqrt(2/M)*np.cos((test_imgs.dot(w.T)+u[np.newaxis,:]))
        test_imgs = test_imgs - mu

        reconstruct_imgs = u_q.dot(u_q.T).dot(test_imgs.T).T
        scores_recon = - np.linalg.norm(test_imgs-reconstruct_imgs, ord=2, axis=1)

        if batch_i == 0:
            all_scores_recon = scores_recon
            all_labels = batch_labels[1].detach().cpu().numpy()
        else:
            all_scores_recon = np.concatenate((all_scores_recon, scores_recon),
                                        axis=0)
            all_labels = np.concatenate((all_labels, 
                                         batch_labels[1].detach().cpu().numpy()),
                                         axis=0)

        gc.collect()
        batch_i += 1

    return all_scores_recon, all_labels

#%%
if kpca_algo == 'CoP':
    train_x, train_y = generate_reconstruction_errors(u_q, 
                                                  classifier_train_dataloader,
                                                  mu=mu,
                                                  model=model,
                                                  kpca_algo=kpca_algo)

if kpca_algo == 'CoRP':
    train_x, train_y = generate_reconstruction_errors(u_q, 
                                                  classifier_train_dataloader,
                                                  mu=mu,
                                                  model=model,
                                                  kpca_algo=kpca_algo,
                                                  w=w,
                                                  u=u,
                                                  M=M)

#%%
# train logistic regression
from sklearn.linear_model import LogisticRegression

train_x = train_x.reshape((-1, 1))

detector_model = LogisticRegression()
detector_model.fit(train_x, train_y)

with open(f'log_reg_{kpca_algo}.pkl', 'wb') as file:
    pickle.dump(detector_model, file)

#%%
# check performance on training data
detector_model.score(train_x, train_y)

# %%
# Test the detector
# 1. Load the testing data
# 2. Generate reconstruction errors.
# 3. Classify using trained logistic regression model.
if kpca_algo == 'CoP':
    test_x, true_test_y = generate_reconstruction_errors(u_q, 
                                                  test_dataloader,
                                                  mu=mu,
                                                  model=model,
                                                  kpca_algo=kpca_algo)

if kpca_algo == 'CoRP':
    test_x, true_test_y = generate_reconstruction_errors(u_q, 
                                                  test_dataloader,
                                                  mu=mu,
                                                  model=model,
                                                  kpca_algo=kpca_algo,
                                                  w=w,
                                                  u=u,
                                                  M=M)
    
# %%
test_x = test_x.reshape((-1, 1))
test_acc = detector_model.score(test_x, true_test_y)
print(test_acc)
pred_test_y_prob = detector_model.predict_proba(test_x)

#%%
roc_auc = roc_auc_score(true_test_y, pred_test_y_prob, 
                        multi_class='ovr')
                        #average=None)
print(roc_auc)
#%%
predictions = detector_model.predict(test_x)
cm = confusion_matrix(true_test_y, predictions, labels=[0, 1, 2, 3])
print(cm)

#%%
cm2 = pycm.ConfusionMatrix(actual_vector=true_test_y,
                      predict_vector=predictions)

cm2.print_matrix()
cm2.stat(summary=True)

# %%
i = 0
for imgs, labels in classifier_train_dataloader:
    v = get_feat_vecs(model, imgs)
    l = labels[1]
    if i == 0:
        i+=1
        feat_vecs = v
        feat_l = l
    else:
        feat_vecs = torch.cat((feat_vecs, v), dim=0)
        feat_l = torch.cat((feat_l, l), dim=0)
    
#%%
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(feat_vecs)

#%%
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=feat_l, cmap='viridis', alpha=0.7)
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
#plt.colorbar(scatter, ticks=np.unique(feat_l), label='Target Class')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
legend1.get_texts()[0].set_text('Clean')
legend1.get_texts()[1].set_text('Poltergeist')
legend1.get_texts()[2].set_text('SNAL')
legend1.get_texts()[3].set_text('ESIA')
ax.add_artist(legend1)
# kw = dict(num=4, color=scatter.cmap(0.7), fmt="$ {x:.2f}",
#           func=lambda s: np.sqrt(s/.3)/3)
# legend2 = ax.legend(*scatter.legend_elements(**kw),
#                     loc="lower left", title="Class")
#plt.grid(True)

plt.savefig('tsne-cop.png', bbox='tight', dpi=150)
plt.show()
# %%
