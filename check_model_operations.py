import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from testers import *
from collections import OrderedDict
import numpy as np
from mobilenet_v2_cifar100_exp_30 import MobileNetV2

import counting

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
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
     #       data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def dataParallel_converter(model, model_path):
    """
        convert between single gpu model and molti-gpu model
    """
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    # model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)
 #   model.cuda()

    # torch.save(model.state_dict(), './model/cifar10_vgg16_acc_93.540_3fc_sgd_in_multigpu.pt')

    return model



def read_block_without_shortcut(block, input_size, f_activation='swish'):
    """Reads the operations on a single EfficientNet block.

    Args:
        block: efficientnet_model.MBConvBlock,
        input_shape: int, square image assumed.
        f_activation: str or None, one of 'relu', 'swish', None.

    Returns:
        list, of operations.
    """
    ops = []

    l_name = 'conv1'
    layer = getattr(block, l_name)
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
  # Input size might have changed.
    input_size = counting.get_conv_output_size(
        image_size=input_size, filter_size=layer_temp.kernel_shape[2],
        padding=layer_temp.padding, stride=layer_temp.strides[0])

    l_name = 'conv2'
    layer = getattr(block, l_name)
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
    # Input size might have changed.
    input_size = counting.get_conv_output_size(
        image_size=input_size, filter_size=layer_temp.kernel_shape[2],
        padding=layer_temp.padding, stride=layer_temp.strides[0])

    l_name = 'conv3'
    layer = getattr(block, l_name)
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
    # Input size might have changed.
    input_size = counting.get_conv_output_size(
        image_size=input_size, filter_size=layer_temp.kernel_shape[2],
        padding=layer_temp.padding, stride=layer_temp.strides[0])


    return ops, input_size


def read_block(block, input_size, f_activation='swish'):
    """Reads the operations on a single EfficientNet block.

    Args:
        block: efficientnet_model.MBConvBlock,
        input_shape: int, square image assumed.
        f_activation: str or None, one of 'relu', 'swish', None.

    Returns:
        list, of operations.
    """
    ops = []

    #shortcut does not need to change the input size
    l_name = 'shortcut'
    layer = getattr(block, l_name)[0]
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))


    l_name = 'conv1'
    layer = getattr(block, l_name)
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
  # Input size might have changed.
    input_size = counting.get_conv_output_size(
        image_size=input_size, filter_size=layer_temp.kernel_shape[2],
        padding=layer_temp.padding, stride=layer_temp.strides[0])

    l_name = 'conv2'
    layer = getattr(block, l_name)
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
    # Input size might have changed.
    input_size = counting.get_conv_output_size(
        image_size=input_size, filter_size=layer_temp.kernel_shape[2],
        padding=layer_temp.padding, stride=layer_temp.strides[0])

    l_name = 'conv3'
    layer = getattr(block, l_name)
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
    # Input size might have changed.
    input_size = counting.get_conv_output_size(
        image_size=input_size, filter_size=layer_temp.kernel_shape[2],
        padding=layer_temp.padding, stride=layer_temp.strides[0])


    return ops, input_size


def read_model(model, input_shape, f_activation='relu'):
    """Reads the operations on a single EfficientNet block.

    Args:
        model: efficientnet_model.Model,
        input_shape: int, square image assumed.
        f_activation: str or None, one of 'relu', 'swish', None.

    Returns:
        list, of operations.
    """
    _ = model(torch.ones(input_shape))
    input_size = input_shape[2]  # Assuming square
    ops = []
  # 1
    l_name = 'conv1'
    layer = getattr(model, l_name)
#  aa = list(layer.weight[0].shape)
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
  # Input size might have changed.
    input_size = counting.get_conv_output_size(
        image_size=input_size, filter_size=layer_temp.kernel_shape[2],
        padding=layer_temp.padding, stride=layer_temp.strides[0])

    for idx, block in enumerate(model.layers):
      #  print(idx)
        if idx == 0 or idx == 1 or idx == 10 or idx == 16:
            block_ops, input_size = read_block(block, input_size, f_activation=f_activation)
            ops.append(('block_%d' % idx, block_ops))
        else:
            block_ops, input_size = read_block_without_shortcut(block, input_size, f_activation=f_activation)
            ops.append(('block_%d' % idx, block_ops))

    l_name = 'conv2'
    layer = getattr(model, l_name)
  #  aa = list(layer.weight[0].shape)
    layer_temp = counting.Conv2D(
        input_size, list(layer.weight.shape), layer.stride,
        layer.padding, True, f_activation)  # Use bias true since batch_norm
    ops.append((l_name, layer_temp))
  # Input size might have changed.
    input_size = counting.get_conv_output_size(
        image_size=input_size, filter_size=layer_temp.kernel_shape[2],
        padding=layer_temp.padding, stride=layer_temp.strides[0])

  # Blocks


  # Head
    l_name = 'linear'
    layer = getattr(model, l_name)
    shapetemp = [layer.in_features, layer.out_features]
    ops.append(('_fc', counting.FullyConnected(
        list(shapetemp), True, None)))

    return ops

def main():
 #   model = VGG(depth=16, init_weights=True, cfg=None)
    model = MobileNetV2()
    model = dataParallel_converter(model, "./cifar100_mobilenetv217_retrained_acc_80.170_config_mobile_v2_0.7_threshold.pt")


    aa = getattr(model, 'conv1')

    input_size = 32
    input_shape = (1, 3, input_size, input_size)

    all_ops = read_model(model, input_shape)
    print('\n'.join(map(str, all_ops)))


    counter = counting.MicroNetCounter(all_ops, add_bits_base=32, mul_bits_base=32)


    INPUT_BITS = 16
    ACCUMULATOR_BITS = 32
    PARAMETER_BITS = 16
    SUMMARIZE_BLOCKS = True

    counter.print_summary(0, PARAMETER_BITS, ACCUMULATOR_BITS, INPUT_BITS, summarize_blocks=SUMMARIZE_BLOCKS)

    counter.print_summary(0.5, PARAMETER_BITS, ACCUMULATOR_BITS, INPUT_BITS, summarize_blocks=SUMMARIZE_BLOCKS)




if __name__ == '__main__':
    main()

