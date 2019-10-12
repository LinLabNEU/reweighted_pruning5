from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from mobilenet_v2_cifar100_exp_30 import MobileNetV2
import argparse
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from numpy import linalg as LA
import sys
import os

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data.cifar100', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])),
    batch_size=256, shuffle=True, **kwargs)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_irregular_sparsity(model):
    """

        :param model: saved re-trained model
        :return:
        """

    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model.named_parameters():
        if len(weight.size()) == 4:
            # continue
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            print(name, non_zeros, zeros, non_zeros+zeros, zeros/(non_zeros+zeros))
            # zeros = np.sum(weight.cpu().detach().numpy() == 0)
            # non_zero = np.sum(weight.cpu().detach().numpy() != 0)
            # print("{}, all weights: {}, irregular zeros: {}, irregular sparsity is: {:.4f}".format(name, zeros+non_zeros, zeros, zeros / (zeros + non_zeros)))
            # print(non_zeros+zeros)
    total_nonzeros += 128000


    # fileObject = open('sampleList.txt', 'w')

    #the_model = torch.load(PATH)
    model.eval()
    # Print model's state_dict

    bn_paras = 0
    #print("Mobilenet_V2 Model's state_dict:")
    for param_tensor in model.state_dict():
        #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        if model.state_dict()[param_tensor].size() != torch.Size([]):
            bn_paras += model.state_dict()[param_tensor].size()[0]
            #print(model.state_dict()[param_tensor].size()[0])
    #print('#########################      v2      ############################')
    #print(model)
    bn_paras = (bn_paras - 200) * 4 /5

    print("---------------------------------------------------------------------------")
    print("total weightsv(fully connected and conv layers): {}, total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros + total_nonzeros, total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("total parameters (fully connected, conv layers and their batchnorm layers) is: {}".format(bn_paras + total_zeros + total_nonzeros))
    print("total bitmask number(32bit) is: {}".format(
        (total_zeros+total_nonzeros) / 32))
    print("batchnorm parameters number is: {}".format(bn_paras))
    print("total non-zero parameters is: {}".format(bn_paras + total_nonzeros))
    print("total parameters for storage is: {}".format((bn_paras + total_nonzeros)/2 + (total_zeros+total_nonzeros) / 32))
    print("===========================================================================\n\n")


def dataParallel_converter(model, model_path):
    # """
    #     convert between single gpu model and molti-gpu model
    # """
    # state_dict = torch.load(model_path)
    # new_state_dict = OrderedDict()

    # for k, v in state_dict.items():
    #     k = k.replace('module.', '')
    #     new_state_dict[k] = v

    # # model = torch.nn.DataParallel(model)
    # model.load_state_dict(new_state_dict)
    ################
    ## single GPU ##
    ################

    # model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(original_model_name).items()})
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] 
        # new_state_dict[name] = v
        # if 'module' not in k:
        #     k = 'module.'+ k
        # else:
        #     k = k.replace('features.module.', 'module.features.')
        k = k.replace('module.', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)  # need basline model

    ################
    ## multi GPUs ##
    ################

    # cudnn.benchmark = True
    # model.load_state_dict(torch.load(original_model_name))
    model.cuda()

    # torch.save(model.state_dict(), './model/cifar10_vgg16_acc_93.540_3fc_sgd_in_multigpu.pt')

    return model


def main():

    model = MobileNetV2()
    model = dataParallel_converter(model, "./cifar100_mobilenetv217_retrained_acc_80.170_config_mobile_v2_0.7_threshold.pt")


    # for name, weight in model.named_parameters():
    #     if (len(weight.size()) == 4):
    #         print(name, weight)
    # print("\n------------------------------\n")

    test(model, device, test_loader)
    test_irregular_sparsity(model)


if __name__ == '__main__':
    main()
