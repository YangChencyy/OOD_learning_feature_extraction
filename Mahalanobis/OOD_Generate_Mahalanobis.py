"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
# import data_loader
import numpy as np
# import calculate_log as callog
# import models
import os
# import lib_generation

from torchvision import transforms
from torch.autograd import Variable

import Mahalanobis.models
import Mahalanobis.lib_generation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Generate_Maha(model, outf, InD_Dataset, OOD_Dataset,
                  trloader, tsloader, OOD_loader, net = 'densenet', gpu = 0, num_classes = 10):

    torch.cuda.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    model.cuda()
    
    # load dataset
    train_loader, test_loader = trloader, tsloader
    
    # set information about feature extaction
    model.eval()
    if InD_Dataset == 'Cifar_10':
        temp_x = torch.rand(2,3,32,32).to_device()
    else:
        temp_x = torch.rand(2,1,28,28).to_device()
    
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    sample_mean, precision = Mahalanobis.lib_generation.sample_estimator(model, num_classes, feature_list, train_loader)
    
    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            M_in = Mahalanobis.lib_generation.get_Mahalanobis_score(model, test_loader, num_classes, outf, \
                                                        True, net, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            

        out_test_loader = OOD_loader
        for i in range(num_output):
            M_out = Mahalanobis.lib_generation.get_Mahalanobis_score(model, out_test_loader, num_classes, outf, \
                                                            False, net, sample_mean, precision, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_data, Mahalanobis_labels = Mahalanobis.lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
        file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), InD_Dataset , OOD_Dataset))
        Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(file_name, Mahalanobis_data)
 

if __name__ == '__main__':
    None