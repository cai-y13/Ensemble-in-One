import torch
import torch.nn as nn
from collections import OrderedDict

def conv2d_template(hps):
    conv2d_str = 'nn.Conv2d({}, {}, kernel_size={}, stride={}, padding={}, dilation={}, bias={}, groups={})'\
            .format(hps['in_channels'], \
            hps['out_channels'], hps['kernel_size'], hps['stride'], \
            hps['padding'], hps['dilation'], hps['bias_term'], hps['groups'])
    return conv2d_str

def conv_transpose2d_template(hps):
    conv_transpose2d_str = 'nn.ConvTranspose2d({}, {}, kernel_size={}, stride={}, padding={}, dilation={}, bias={}, groups={})'\
            .format(hps['in_channels'], \
            hps['out_channels'], hps['kernel_size'], hps['stride'], \
            hps['padding'], hps['dilation'], hps['bias_term'], hps['groups'])
    return conv_transpose2d_str

def linear_template(hps):
    linear_str = 'nn.Linear({}, {}, bias={})'.format(hps['in_features'], \
            hps['out_features'], hps['bias_term'])
    return linear_str

def batchnorm_template(hps):
    batchnorm_str = 'nn.BatchNorm2d({})'.format(hps['num_features'])
    return batchnorm_str

def relu_template():
    relu_str = 'nn.ReLU(inplace=True)'
    return relu_str

def relu6_template():
    relu6_str = 'nn.ReLU6(inplace=True)'
    return relu6_str

def sigmoid_template():
    sigmoid_str = 'nn.Sigmoid()'
    return sigmoid_str

def leaky_relu_template(hps):
    leaky_relu_str = 'nn.LeakyReLU({}, inplace=True)'.format(hps['negative_slope'])
    return leaky_relu_str

def pool_template(hps):
    func = 'nn.AvgPool2d' if hps['subtype'] == 'avgpool' else 'nn.MaxPool2d'
    pool_str = '{}({}, stride={}, padding={})'.format(func, \
            hps['kernel_size'], hps['stride'], hps['padding'])
    return pool_str

def const_template(const_shape):
    const_str = 'ConstFunc({})'.format(str(const_shape))
    return const_str


def derive_model(net):
    dfn = []
    gen_type = {
            'conv2d': conv2d_template,
            'deconv': conv_transpose2d_template,
            'linear': linear_template, 
            'relu': relu_template, 
            'relu6': relu6_template,
            'leaky_relu': leaky_relu_template,
            'sigmoid': sigmoid_template,
            'batchnorm': batchnorm_template,
            'pool': pool_template,
            'const': const_template,
    }
    for i in range(len(net)):
        layer = net[i]
        layer_type = layer['type']
        if layer_type in ['add', 'mul'] and layer['const'] == True:
            template = gen_type['const']
            dfn_layer = 'self.{} = '.format(layer['name'] + '_const') + template(list(layer['const_value'].shape))
            dfn.append(dfn_layer)
            continue
            
        if layer_type not in gen_type:
            continue
        template = gen_type[layer_type]
        if layer_type not in ['relu', 'relu6', 'sigmoid']:
            if layer_type == 'batchnorm' and layer['activate'] == False:
                continue
            dfn_layer = template(layer['hps'])
        else:
            dfn_layer = template()
        prefix = 'self.{} = '.format(layer['name'])
        dfn_layer = prefix + dfn_layer
        dfn.append(dfn_layer)
    return dfn

def add_forward_template(bottom, top):
    assert len(bottom) == 2, 'currently only support two bottom blobs adding'
    return '{} = {} + {}'.format(top, bottom[0], bottom[1])

def mul_forward_template(bottom, top):
    assert len(bottom) == 2, 'currently only support two bottom blobs multiplication'
    return '{} = {} * {}'.format(top, bottom[0], bottom[1])

def view_forward_template(bottom, top, new_shape):
    new_shape[0] = bottom+'.shape[0]'
    new_shape = tuple(new_shape)
    new_shape = str(new_shape).replace("'", '')
    return '{} = {}.view{}'.format(top, bottom, new_shape)

def concat_forward_template(bottoms, top, cat_dim):
    concat_fw_str = top + ' = ' + 'torch.cat(['
    count = 0
    for bottom in bottoms:
        concat_fw_str += bottom
        if count < len(bottoms) - 1:
            concat_fw_str += ', '
        else:
            concat_fw_str += '], dim={})'.format(cat_dim)
        count += 1
    return concat_fw_str

def split_forward_template(bottom, tops, split_size, split_dim):
    split_fw_str = ''
    count = 0
    for top in tops:
        split_fw_str += top
        if count < len(tops) - 1:
            split_fw_str += ', '
        else:
            split_fw_str += ' '
        count += 1
    split_fw_str += '= torch.split({}, {}, {})'.format(bottom, split_size, split_dim)
    return split_fw_str

def upsample_forward_template(bottom, top, scale):
    return '{} = F.interpolate({}, scale_factor=[{}, {}], mode=\'nearest\')'.format(top, bottom, scale[0], scale[1])

def exp_forward_template(bottom, top):
    return '{} = torch.exp({})'.format(top, bottom)

def inv_forward_template(bottom, top):
    return '{} = 1.0 / {}'.format(top, bottom)

def contiguous_forward_template(bottom, top):
    return '{} = {}.contiguous()'.format(top, bottom)

def permute_forward_template(bottom, top, new_dim):
    return '{} = {}.permute({}, {}, {}, {})'.format(top, bottom, new_dim[0], new_dim[1], new_dim[2], new_dim[3])

def pow_forward_template(bottom, top, exponent):
    return '{} = torch.pow({}, {})'.format(top, bottom, exponent)

def derive_forward(net):
    forward = []
    for i in range(len(net)):
        layer = net[i]
        layer_type = layer['type']
        bottom = layer['bottom']
        top = layer['top']
        if layer_type not in ['add', 'mul', 'view', 'concat', 'split', 'upsample', 'contiguous', 'permute', 'exp', 'inv']:
            if layer_type == 'batchnorm' and layer['activate'] == False:
                continue
            cmd = '{} = self.{}({})'.format(top, layer['name'], bottom)
        elif layer_type == 'add':
            if layer['const'] == True:
                for b in bottom:
                    if 'const' in b:
                        const_bottom = b
                cmd = '{} = self.{}_const()'.format(const_bottom, layer['name'])
                forward.append(cmd)
            cmd = add_forward_template(bottom, top)
        elif layer_type == 'mul':
            if layer['const'] == True:
                for b in bottom:
                    if 'const' in b:
                        const_bottom = b
                cmd = '{} = self.{}_const()'.format(const_bottom, layer['name'])
                forward.append(cmd)
            cmd = mul_forward_template(bottom, top)
        elif layer_type == 'view':
            cmd = view_forward_template(bottom, top, layer['new_shape'])
        elif layer_type == 'concat':
            cmd = concat_forward_template(bottom, top, layer['cat_dim'])
        elif layer_type == 'split':
            cmd = split_forward_template(bottom, top, layer['split_size'], layer['split_dim'])
        elif layer_type == 'upsample':
            cmd = upsample_forward_template(bottom, top, (layer['height_factor'], layer['width_factor']))
        elif layer_type == 'contiguous':
            cmd = contiguous_forward_template(bottom, top)
        elif layer_type == 'permute':
            cmd = permute_forward_template(bottom, top, layer['permute_dim'])
        elif layer_type == 'exp':
            cmd = exp_forward_template(bottom, top)
        elif layer_type == 'inv':
            cmd = inv_forward_template(bottom, top)
        elif layer_type == 'pow':
            cmd = pow_forward_template(bottom, top, layer['hps']['exponent'])
        forward.append(cmd)
    return forward

def derive_checkpoint(net):
    new_state_dict = OrderedDict()
    for i in range(len(net)):
        layer = net[i]
        layer_name = layer['name']
        layer_type = layer['type']
        has_param_types = {
            'conv2d': ['weight', 'bias'],
            'deconv': ['weight', 'bias'],
            'linear': ['weight', 'bias'],
            'batchnorm': ['weight', 'bias', 'running_mean', 'running_var'],
        }
        if layer_type == 'add' or layer_type == 'mul':
            if layer['const'] == True:
                param_key = '{}.{}'.format(layer_name + '_const', 'const_value')
                new_state_dict[param_key] = torch.tensor(layer['const_value'])
        if layer_type in has_param_types:
            params = has_param_types[layer_type]
            if layer_type == 'batchnorm' and layer['activate'] == False:
                continue
            for param_id in range(len(params)):
                param_type = params[param_id]
                if layer[param_type] is not None:
                    param_key = '{}.{}'.format(layer_name, param_type)
                    new_state_dict[param_key] = torch.tensor(layer[param_type])
    return new_state_dict

