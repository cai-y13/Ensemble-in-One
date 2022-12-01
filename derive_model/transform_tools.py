import torch
import torch.nn as nn
import numpy as np
import sys
import os
from derive_model.extract_model_arch import extract_arch
from derive_model.merge_conv_bn import net_merge_conv_bn
from derive_model.derive import derive_model, derive_forward, derive_checkpoint
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help='The defination file of the model to transform')
    parser.add_argument('--weight_file', type=str, default=None, help='The weight file of the model to transform')
    parser.add_argument('--model_name', type=str, help='The name of the model class')
    parser.add_argument('--merge_conv_bn', type=bool, default=False, help='If merging the convolution and batchnorm')

    args = parser.parse_args()
    
    return args

def construct_model(net_arch, return_blobs, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    space = '    '
    dfn_head = [
        'import torch\n',
        'import torch.nn as nn\n',
        'import torch.nn.functional as F\n',
        'from dpt.utils.function import *\n',
        'class derived_model(nn.Module):\n',
        space + 'def __init__(self):\n',
        space * 2 + 'super(derived_model, self).__init__()\n'
    ]
    output_net_file = os.path.join(output_dir, 'derived_model.py')
    net_file = open(output_net_file, 'w')
    for line in range(len(dfn_head)):
        net_file.write(dfn_head[line])

    dfn_arch = derive_model(net_arch)
    prefix = space * 2
    for line in range(len(dfn_arch)):
        net_file.write(prefix + dfn_arch[line] + '\n')

    forward_head = space + 'def forward(self, {}):'.format(net_arch[0]['bottom']) + '\n'
    net_file.write(forward_head)
    forward_cmd = derive_forward(net_arch)
    for line in range(len(forward_cmd)):
        net_file.write(prefix + forward_cmd[line] + '\n')
    forward_end = prefix + 'return '
    for i, return_blob in enumerate(return_blobs):
        forward_end += return_blob
        if i < len(return_blobs) - 1:
            forward_end += ', '
    forward_end += '\n'
    #forward_end = prefix + 'return {}'.format(net_arch[-1]['top']) + '\n'
    net_file.write(forward_end)
    net_file.close()

def construct_weight(net_arch, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    new_state_dict = derive_checkpoint(net_arch)
    weight_file = os.path.join(output_dir, 'derived_weight.pth')
    torch.save(new_state_dict, weight_file)


if __name__ == '__main__':
    model_import = args.model_file.replace('/', '.').replace('.py', '')
    exec('from {} import {}'.format(model_import, args.model_name))

    model = eval(args.model_name)
    net = model()
    if args.weight_file is not None:
        weight = torch.load(args.weight_file)
        net.load_state_dict(weight)

    input_var = torch.randn(1, 3, 224, 224)
    #input_var = torch.randn(1, 1, 800, 800)
    #input_var = torch.rand(1, 1, 17, 1)

    arch = extract_arch(net, input_var)
    np.save('output/arch.npy', arch)

    if args.merge_conv_bn:
        arch = net_merge_conv_bn(arch)

    construct_model(arch)
    construct_weight(arch)
