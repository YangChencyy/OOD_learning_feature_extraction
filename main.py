import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import time
import os
# import sys

import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import umap

from dataset import *

from Multi_GP.multi_GP import *
from Multi_GP.model_cifar import Cifar_10_Net, BasicBlock, resnet18, load_part



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = 0

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
    'Cifar_10': Cifar_10_Net   
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GP parameters")

    # Add a positional argument for the number
    parser.add_argument("InD_Dataset", type=str, help="The name of the InD dataset.")
    parser.add_argument("train_batch_size", type=int, help="train_batch_size")
    parser.add_argument("test_batch_size", type=int, help="test_batch_size")

    args = parser.parse_args()
    print("InD dataset:", args.InD_Dataset)

    
    num_classes = 10

    train_set, test_set, trloader, tsloader = data_dic[args.InD_Dataset](batch_size = args.train_batch_size, 
                                                                    test_batch_size = args.test_batch_size)
    OOD_sets, OOD_loaders = [], []
    grey = True
    if args.InD_Dataset == 'Cifar_10':
        OOD_Dataset = ['SVHN', 'Imagenet_r', 'Imagenet_c']
        grey = False
    else:
        if args.InD_Dataset == 'MNIST':
            OOD_Dataset = ['FashionMNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']
        elif args.InD_Dataset == 'FashionMNIST':
            OOD_Dataset = ['MNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']

    # Get all OOD datasets     
    for dataset in OOD_Dataset:
        _, OOD_set, _, OODloader = data_dic[dataset](batch_size = args.train_batch_size, 
                                                test_batch_size = args.test_batch_size, into_grey = grey)
        OOD_sets.append(OOD_set)
        OOD_loaders.append(OODloader)

    # mkdir directory to save
    parent_dir = os.getcwd()
    directory = 'Multi_GP/store_data/' + args.InD_Dataset
    path = os.path.join(parent_dir, directory)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    # Get all labels of training data for GP
    InD_label = []
    for i in range(len(train_set)):
        InD_label.append(train_set.__getitem__(i)[1])

    net = None
    if args.InD_Dataset == 'Cifar_10':
        # pretrained_resnet18 = resnet18(pretrained=True)
        net = Cifar_10_Net(BasicBlock, [2, 2, 2, 2], dim_f = 32)
        # network.load_sta(torch.load('path'))
        # net = load_part(net, pretrained_resnet18.state_dict())

        cifar10_train(network = net, trloader = trloader, epochs = 30, optim = 'SGD', verbal=True)
        torch.save(net.state_dict(), os.path.join(parent_dir, args.InD_Dataset + "_net.pt"))
        # torch.save(net, os.path.join(parent_dir, InD_Dataset + "_net.pt"))

    else:
        epochs = 4
        net = data_model[args.InD_Dataset]()
        train(network = net, trloader = trloader, epochs = epochs, verbal=True)
        # torch.save(net, os.path.join(parent_dir, InD_Dataset + "_net.pt"))
        torch.save(net.state_dict(), os.path.join(parent_dir, args.InD_Dataset + "_net.pt"))


    ## get InD data for GP
    InD_features, InD_scores, InD_acc = scores(net, trloader)
    # test_feature, test_score, test_acc = scores(net, tsloader)
    print("InD accuracy: ", InD_acc)
    InD_feature, InD_score = InD_features[0:20000], InD_scores[0:20000]
    # test_feature, test_score = test_feature[0:5000], test_score[0:5000]
    test_feature, test_score = InD_features[20000:25000], InD_scores[20000:25000]
    labels = train_set.targets[20000:25000] # .numpy().tolist() 

    train_data = np.concatenate((InD_feature.cpu().numpy(), InD_score.cpu().numpy()), 1)
    train_data = pd.DataFrame(train_data)
    train_data['label'] = InD_label[0:20000]
    train_data.to_csv(directory +  '/train.csv')
    print("train data stored")

    ## get OOD data for GP
    for i in range(len(OOD_loaders)):

        OOD_feature, OOD_score = scoresOOD(net, OOD_loaders[i])
        OOD_feature, OOD_score = OOD_feature[0:5000], OOD_score[0:5000]


        total_CNN = np.concatenate((test_feature.cpu().numpy(), OOD_feature.cpu().numpy()), 0)
        reducer_CNN = umap.UMAP(random_state = 42, n_neighbors=10, n_components=2)
        UMAPf = reducer_CNN.fit_transform(total_CNN)

        # all_feature = np.concatenate((test_feature, OOD_feature), 0)
        all_score = np.concatenate((test_score.cpu().numpy(), OOD_score.cpu().numpy()), 0)
        DNN_data = np.concatenate((total_CNN, all_score, UMAPf), 1)

        data_df = pd.DataFrame(DNN_data) 
        data_df['class'] = ['test']*len(test_feature) + ['OOD']*len(OOD_feature)
        # print(type(labels))
        data_df['label'] = labels + [10]*len(OOD_feature)

        data_df.to_csv(directory +  '/' + OOD_Dataset[i] + '_test.csv')
        print(OOD_Dataset[i] + " test data stored")


    print("\nEND")









