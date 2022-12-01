import torch
import torch.nn as nn
import numpy as np

def merge_conv_bn_param(conv, bn):
    # convolution parameters
    W = conv['weight']
    bias = conv['bias'] if conv['hps']['bias_term'] else np.zeros(W.shape[0])

    # batchnorm parameters
    mu = bn['running_mean']
    var = bn['running_var']
    gamma = bn['weight']
    beta = bn['bias']

    eps = bn['eps']
    denom = np.sqrt(var+eps)
    b = beta - gamma * mu / denom
    A = gamma / denom
    bias = bias * A

    A = np.expand_dims(np.expand_dims(np.expand_dims(A, 1), 2), 3)
    W = W * A 
    bias = b + bias

    conv['weight'] = W
    conv['bias'] = bias

    bn['activate'] = False
    conv['hps']['bias_term'] = True
    #conv['top'] = bn['top']


def net_merge_conv_bn(net):
    # Scan the layers
    for l in range(len(net)):
        layer = net[l]
        if layer['type'] not in ['conv2d', 'deconv2d']:
            continue
        top_name = layer['top']
        # inner scan to search if there is a following batchnorm layer
        for i in range(len(net)):
            layer_inner = net[i]
            if (layer_inner['bottom'] == top_name and layer_inner['type'] == 'batchnorm'):
                merge_conv_bn_param(layer, layer_inner)
                for j in range(len(net)):
                    layer_bn_next = net[j]
                    if (isinstance(layer_bn_next['bottom'], str) and layer_bn_next['bottom'] == layer_inner['top']):
                        layer_bn_next['bottom'] = top_name
                    elif (isinstance(layer_bn_next['bottom'], list)):
                        new_bottom = []
                        for bottom in layer_bn_next['bottom']:
                            if bottom == layer_inner['top']:
                                new_bottom.append(top_name)
                            else:
                                new_bottom.append(bottom)
                        layer_bn_next['bottom'] = new_bottom
                break
    return net


