from dataset import Cifar_100_dataset
from Multi_GP.model_cifar import Cifar_10_Net, BasicBlock
from Multi_GP.multi_GP import *

import os

f_size = 32
train_set, test_set, trloader, tsloader = Cifar_100_dataset(batch_size = 128, test_batch_size = 128)

net = Cifar_10_Net(BasicBlock, [2, 2, 2, 2], num_classes = 100, dim_f = f_size)
cifar10_train(network = net, trloader = trloader, epochs = 20, optim = 'SGD', verbal=True)
_, _, train_acc = score_new(net, trloader)
print("Train accuracy: ", train_acc)
_, _, test_acc = score_new(net, trloader)
print("Train accuracy: ", train_acc)
