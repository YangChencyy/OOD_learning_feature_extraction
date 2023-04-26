
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
torch.manual_seed(424)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import umap

from dataset import *
from multi_GP import * 




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epochs = 5
train_batch_size = 64
test_batch_size = 64
InD_Dataset = 'MNIST'
OOD_Dataset = ['FashionMNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']

data_dic = {
    'MNIST': MNIST_dataset,
    'FashionMNIST': Fashion_MNIST_dataset, 
    'Cifar_10': Cifar_10_dataset,
    'SVHN': SVHN_dataset, 
    'Imagenet_r': TinyImagenet_r_dataset,
    'Imagenet_c': TinyImagenet_c_dataset
}


data_model = {
    'MNIST': MNIST_Net,
    'FashionMNIST': Fashion_MNIST_Net, 
    'Cifar10': Cifar_10_Net   
}


_, test_set, trloader, tsloader = data_dic[InD_Dataset](batch_size = train_batch_size, test_batch_size = test_batch_size)
OOD_loaders = []
for dataset in OOD_Dataset:
    _, _, _, OODloader = data_dic[dataset](batch_size = train_batch_size, test_batch_size = test_batch_size)
    OOD_loaders.append(OODloader)


# multi_GP
net = data_model[InD_Dataset]()
train(network = net, trloader = trloader, epochs = epochs)

## get InD data
InD_feature, InD_score, acc = scores(net, tsloader)

## get OOD data
OOD_features_dict = dict(zip(OOD_Dataset, [list()]*len(OOD_Dataset)))
for i in len(OOD_loaders):
    print("OOD", i)
    feature, score = scoresOOD(net, OOD_loaders[i])
    OOD_features_dict[OOD_Dataset[i]].append(feature)
    OOD_features_dict[OOD_Dataset[i]].append(score)

    total_CNN = np.concatenate((InD_feature, feature), 0)
    reducer_CNN = umap.UMAP(random_state = 42, n_neighbors=10, n_components=2)
    UMAPf = reducer_CNN.fit_transform(total_CNN)








# DUQ





# SNGP





# Mahalanobis





# Temperature scaling

