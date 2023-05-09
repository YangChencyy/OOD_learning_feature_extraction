
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

from DUQ.train_duq_fm import train_model
from DUQ.evaluate_ood import get_auroc_ood


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

methods = [2]
epochs = 5
train_batch_size = 64
test_batch_size = 64
InD_Dataset = 'MNIST'
parent_dir = os.getcwd()
directory = 'store_data/' + InD_Dataset
path = os.path.join(parent_dir, directory)
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!")

OOD_Dataset = ['FashionMNIST'] #, 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']

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


train_set, test_set, trloader, tsloader = data_dic[InD_Dataset](batch_size = train_batch_size, test_batch_size = test_batch_size)
# Get all labels of training data for GP
InD_label = []
for i in range(len(train_set)):
    InD_label.append(train_set.__getitem__(i)[1])
print(len(InD_label))

# Get all OOD datasets
OOD_sets, OOD_loaders = [], []
for dataset in OOD_Dataset:
    _, OOD_set, _, OODloader = data_dic[dataset](batch_size = train_batch_size, test_batch_size = test_batch_size)
    OOD_sets.append(OOD_set)
    OOD_loaders.append(OODloader)


# multi_GP
if 1 in methods:
    print("Method 1: Multi-GP")
    net = data_model[InD_Dataset]()
    train(network = net, trloader = trloader, epochs = epochs)

    ## get InD data for GP
    InD_feature, InD_score, InD_acc = scores(net, trloader)
    test_feature, test_score, test_acc = scores(net, tsloader)
    print("InD accuracy: ", InD_acc)
    InD_feature, InD_score = InD_feature[0:20000], InD_score[0:20000]
    test_feature, test_score = test_feature[0:5000], test_score[0:5000]


    ## get OOD data for GP
    for i in range(len(OOD_loaders)):

        OOD_feature, OOD_score = scoresOOD(net, OOD_loaders[i])
        OOD_feature, OOD_score = OOD_feature[0:5000], OOD_score[0:5000]


        total_CNN = np.concatenate((test_feature, OOD_feature), 0)
        reducer_CNN = umap.UMAP(random_state = 42, n_neighbors=10, n_components=2)
        UMAPf = reducer_CNN.fit_transform(total_CNN)

        all_feature = np.concatenate((test_feature, OOD_feature), 0)
        all_score = np.concatenate((test_score, OOD_score), 0)
        DNN_data = np.concatenate((all_feature, all_score, UMAPf), 1)

        data_df = pd.DataFrame(DNN_data) 
        data_df['class'] = ['test']*len(test_feature) + ['OOD']*len(OOD_feature)

        data_df.to_csv(directory +  '/' + OOD_Dataset[i] + 'cifar_test.csv')




# DUQ
if 2 in methods:
    print("Method 2: DUQ")
    l_gradient_penalties = [0.0]
    length_scales = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    repetition = 1  # Increase for multiple repetitions
    final_model = False  # set true for final model to train on full train set

    results = {}


    for l_gradient_penalty in l_gradient_penalties:
        for length_scale in length_scales:
            val_accuracies = []
            test_accuracies = []
            roc_aucs_mnist = []
            roc_aucs_notmnist = []

            for _ in range(repetition):
                print(" ### NEW MODEL ### ")
                model, val_accuracy, test_accuracy = train_model(
                    l_gradient_penalty, length_scale, final_model
                )
                # accuracy, roc_auc_mnist = get_fashionmnist_mnist_ood(model)
                # _, roc_auc_notmnist = get_fashionmnist_notmnist_ood(model)

                for ood_set in OOD_sets:
                    accuracy, roc_auc_mnist = get_auroc_ood(test_set, ood_set, model)

                    val_accuracies.append(val_accuracy)
                    test_accuracies.append(test_accuracy)
                    roc_aucs_mnist.append(roc_auc_mnist)
                #roc_aucs_notmnist.append(roc_auc_notmnist)

            results[f"lgp{l_gradient_penalty}_ls{length_scale}"] = [
                (np.mean(val_accuracies), np.std(val_accuracies)),
                (np.mean(test_accuracies), np.std(test_accuracies)),
                (np.mean(roc_aucs_mnist), np.std(roc_aucs_mnist)),
                (np.mean(roc_aucs_notmnist), np.std(roc_aucs_notmnist)),
            ]
            print(results[f"lgp{l_gradient_penalty}_ls{length_scale}"])

    print(results)






# SNGP





# Mahalanobis





# Temperature scaling

