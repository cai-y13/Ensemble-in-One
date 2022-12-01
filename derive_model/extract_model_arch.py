import torch
import torch.nn as nn
import traceback
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import numpy as np

"""
How to support a new layer type:
 layer_name=log.add_layer(layer_type_name)
 top_blobs=log.add_blobs(<output of that layer>)
 layer=caffe_net.Layer_param(xxx)
 <set layer parameters>
 [<layer.add_data(*datas)>]
 log.net.add_layer(layer)
 
Please MUTE the inplace operations to avoid not find in graph

"""

# TODO: support the inplace output of the layers

class Blob_LOG():
    def __init__(self):
        self.data={}
    def __setitem__(self, key, value):
        self.data[key]=value
    def __getitem__(self, key):
        return self.data[key]
    def __len__(self):
        return len(self.data)

NET_INITTED=False

class TransLog(object):
    def __init__(self):
        """
        doing init() with inputs Variable before using it
        """
        self.layers = {}
        self.detail_layers = {}  
        self.detail_blobs = {}  
        self._blobs = Blob_LOG()
        self._blobs_data = []
        self.net = [] 
        self.debug = False #True

    def init(self, inputs):
        """
        :param inputs: is a list of input variables
        """
        self.add_blobs(inputs)

    def reset(self):
        self.layers = {}
        self.detail_layers = {}
        self.detail_blobs = {}
        self._blobs = Blob_LOG()
        self._blobs_data = []
        self.net = []
        self.debug = False

    def add_layer(self, name='layer'):
        if name in self.layers:
            return self.layers[name]
        if name not in self.detail_layers.keys():
            self.detail_layers[name] =0
        self.detail_layers[name] += 1
        name='{}{}'.format(name, self.detail_layers[name])
        self.layers[name] = name
        if self.debug:
            print("{} was added to layers".format(self.layers[name]))
        return self.layers[name]

    def add_blobs(self, blobs, name='blob', with_num=True):
        rst=[]
        for blob in blobs:
            self._blobs_data.append(blob) # to block the memory address be rewrited
            blob_id = int(id(blob))
            if name not in self.detail_blobs.keys():
                self.detail_blobs[name] = 0
            self.detail_blobs[name] += 1           
            if with_num:
                rst.append('{}{}'.format(name, self.detail_blobs[name]))
            else:
                rst.append('{}'.format(name))
            if self.debug:
                print("{}:{} was added to blobs".format(blob_id, rst[-1]))
            self._blobs[blob_id] = rst[-1]
        return rst

    def blobs(self, var, allow_unfound=False):
        var=id(var)
        try:
            return self._blobs[var]
        except:
            if allow_unfound:
                return None
            else:
                print("WARNING: CANNOT FOUND blob {}".format(var))
                #print(self.net)
                raise TypeError('unknown blob id, there might be unsupported layer')
                return None

log = TransLog()

layer_names = {}

def _conv2d(raw,input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    x=raw(input, weight, bias, stride, padding, dilation, groups)
    if not NET_INITTED:
        return x
    name = log.add_layer(name='conv')
    log.add_blobs([x],name='conv_blob')
    layer = {}
    layer['name'] = name
    layer['type'] = 'conv2d'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    if log.blobs(weight, allow_unfound=True) is None:
        log.add_blobs([weight], name='conv_weight')
    else:
        fake_weight = weight.clone()
        log.add_blobs([fake_weight], name='conv_weight')
    layer['weight_name'] = log.blobs(weight)
    layer['weight'] = weight.data.cpu().numpy()
    hps = {}
    hps['stride'] = _pair(stride) if isinstance(stride, int) else stride
    hps['padding'] = _pair(padding) if isinstance(padding, int) else padding
    hps['dilation'] = _pair(dilation) if isinstance(dilation, int) else dilation
    hps['groups'] = groups
    hps['out_channels'] = weight.shape[0]
    hps['in_channels'] = weight.shape[1] * hps['groups']
    hps['kernel_size'] = (weight.shape[2], weight.shape[3])
    hps['deconv_flag'] = 'nodeconv'
    hps['deconv_height_factor'] = 1
    hps['deconv_width_factor'] = 1
    hps['height_factor'] = 1
    hps['width_factor'] = 1
    if bias is not None:
        layer['bias'] = bias.data.cpu().numpy()
        hps['bias_term'] = 1 #True
    else:
        layer['bias'] = None
        hps['bias_term'] = 0 #False
    layer['hps'] = hps
    log.net.append(layer)
    return x

def _conv_transpose2d(raw, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    if not NET_INITTED:
        return x
    name = log.add_layer(name='deconv')
    log.add_blobs([x], name='deconv_blob')
    assert output_padding == 0 or output_padding == (0, 0)
    layer = {}
    layer['name'] = name
    layer['type'] = 'deconv'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    layer['weight'] = weight.data.cpu().numpy()
    hps = {}
    hps['stride'] = _pair(stride) if isinstance(stride, int) else stride
    hps['padding'] = _pair(padding) if isinstance(padding, int) else padding
    hps['dilation'] = _pair(dilation) if isinstance(dilation, int) else dilation
    hps['groups'] = groups
    hps['out_channels'] = weight.shape[1] * hps['groups']
    hps['in_channels'] = weight.shape[0]
    hps['kernel_size'] = (weight.shape[2], weight.shape[3])
    if bias is not None:
        layer['bias'] = bias.data.cpu().numpy()
        hps['bias_term'] = 1
    else:
        layer['bias'] = None
        hps['bias_term'] = 0
    layer['hps'] = hps
    log.net.append(layer)
    return x

def _linear(raw,input, weight, bias=None):
    x = raw(input, weight, bias)
    if not NET_INITTED:
        return x
    layer_name=log.add_layer(name='fc')
    top_blobs=log.add_blobs([x],name='fc_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'linear'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    if log.blobs(weight, allow_unfound=True) is None:
        log.add_blobs([weight], name='fc_weight')
    else:
        fake_weight = weight.clone()
        log.add_blobs([fake_weight], name='fc_weight')
    layer['weight_name'] = log.blobs(weight)
    layer['weight'] = weight.data.cpu().numpy() 
    hps = {}
    hps['out_features'] = weight.shape[0]
    hps['in_features'] = weight.shape[1]
    hps['height_factor'] = 1
    hps['width_factor'] = 1
    if bias is not None:
        layer['bias'] = bias.data.cpu().numpy() 
        hps['bias_term'] = 1 #True
    else:
        layer['bias'] = None
        hps['bias_term'] = 0 #False
    layer['hps'] = hps
    log.net.append(layer)
    return x

def special_avgpool_template(tmp_input, kernel_size, stride, padding, ceil_mode, layer_name, tmp_output=None, register_out=True):
    if tmp_output is None:
        tmp_output = torch.randn(1, tmp_input.shape[1], (tmp_input.shape[2] + 2 * padding[0]) // stride[0], \
                tmp_input.shape[3] // kernel_size[1])
    if register_out:
        top_blobs = log.add_blobs([tmp_output], name=layer_name[:-1] + '_blob')
    if kernel_size[0] <= 8 and kernel_size[1] <= 8 and kernel_size[0] == kernel_size[1]:
        layer = {}
        layer['name'] = layer_name 
        layer['type'] = 'pool'
        layer['bottom'] = log.blobs(tmp_input)
        layer['bottom_shape'] = list(tmp_input.shape[1:])
        layer['top'] = log.blobs(tmp_output)
        layer['top_shape'] = list(tmp_output.shape[1:])
        hps = {}
        hps['kernel_size'] = kernel_size
        hps['stride'] = stride
        hps['padding'] = padding
        hps['ceil_mode'] = ceil_mode
        hps['subtype'] = 'avgpool'
        layer['hps'] = hps
    else:
        layer = {}
        layer['name'] = layer_name 
        layer['type'] = 'conv2d'
        layer['bottom'] = log.blobs(tmp_input)
        layer['top'] = log.blobs(tmp_output)
        layer['bottom_shape'] = list(tmp_input.shape[1:])
        layer['top_shape'] = list(tmp_output.shape[1:])
        weight = np.ones([tmp_input.shape[1], 1, kernel_size[0], kernel_size[1]]) * \
                (1.0 / (kernel_size[0] * kernel_size[1]))
        layer['weight_name'] = layer_name + '_weight'
        layer['weight'] = weight 
        hps = {}
        hps['stride'] = stride 
        hps['padding'] = padding 
        hps['dilation'] = (1, 1) 
        hps['groups'] = tmp_input.shape[1]
        hps['out_channels'] = weight.shape[0]
        hps['in_channels'] = weight.shape[1] * hps['groups']
        hps['kernel_size'] = (weight.shape[2], weight.shape[3])
        hps['deconv_flag'] = 'nodeconv'
        hps['deconv_height_factor'] = 1
        hps['deconv_width_factor'] = 1
        hps['height_factor'] = 1
        hps['width_factor'] = 1
        layer['bias'] = None
        hps['bias_term'] = 0 #False
        layer['hps'] = hps

    return layer, tmp_output


def _pool(type, input, x, kernel_size, stride, padding, ceil_mode):
    kernel_size = _pair(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _pair(stride) if isinstance(stride, int) else stride
    padding = _pair(padding) if isinstance(padding, int) else padding
    if x.shape[2] == 1 and x.shape[3] == 1:
        stride = (1, 1)
    if (type == 'maxpool' and kernel_size[0] <= 8 and kernel_size[1] <= 8) or \
       (type == 'avgpool' and kernel_size[0] <= 8 and kernel_size[1] <= 8 and kernel_size[0] == kernel_size[1]):
        layer_name = log.add_layer(name='{}_pool'.format(type))
        top_blobs = log.add_blobs([x], name='{}_pool_blob'.format(type))
        layer = {}
        layer['name'] = layer_name
        layer['type'] = 'pool'
        layer['bottom'] = log.blobs(input)
        layer['top'] = log.blobs(x)
        layer['bottom_shape'] = list(input.shape[1:])
        layer['top_shape'] = list(x.shape[1:])
        hps = {}
        hps['kernel_size'] = kernel_size 
        hps['stride'] = stride 
        hps['padding'] = padding 
        hps['ceil_mode'] = ceil_mode
        hps['subtype'] = type
        layer['hps'] = hps
        log.net.append(layer)
    else:
        if (kernel_size[0] > 8 or kernel_size[1] > 8) and kernel_size[0] <= 8 and kernel_size[1] <= 8:
            layer_name = log.add_layer(name='{}_pool'.format(type))
            top_blobs = log.add_blobs([x], name='{}_pool_blob'.format(type))
            layer, x = special_avgpool_template(input, kernel_size, stride, padding, ceil_mode, layer_name, x, False)
            log.net.append(layer)
        else:
            # only support global average pooling
            assert x.shape[2] == 1 and x.shape[3] == 1
            remain_h, remain_w = kernel_size[0], kernel_size[1]
            tmp_input = input
            base_layer_name = log.add_layer(name='{}_pool'.format(type))
            fake_blob = torch.randn(x.shape)
            top_blobs = log.add_blobs([fake_blob], name='{}_pool_blob'.format(type))
            while remain_h > 1 or remain_w > 1:
                kernel_h, kernel_w = 1, 1
                can_div_h, can_div_w = False, False
                for div_h in range(8, 1, -1):
                    if remain_h % div_h == 0:
                        kernel_h = div_h
                        can_div_h = True
                        break
                for div_w in range(8, 1, -1):
                    if remain_w % div_w == 0:
                        kernel_w = div_w
                        can_div_w = True
                        break
                assert can_div_h == True and can_div_w == True
                remain_h = remain_h // kernel_h
                remain_w = remain_w // kernel_w
                if remain_h == 1 and remain_w == 1:
                    layer_name = log.add_layer(name=base_layer_name + '_level')
                    layer, _ = special_avgpool_template(tmp_input, (kernel_h, kernel_w), (1, 1), (0, 0), ceil_mode, \
                            layer_name, x)
                    log.net.append(layer)
                else:
                    layer_name = log.add_layer(name=base_layer_name + '_level')
                    layer, tmp_output = special_avgpool_template(tmp_input, (kernel_h, kernel_w), \
                            (kernel_h, kernel_w), (0, 0), ceil_mode, layer_name)
                    tmp_input = tmp_output
                    log.net.append(layer)
                    

def _max_pool2d(raw, input, kernel_size, stride=None, padding=0, dilation=1,
               return_indices=False, ceil_mode=False):
    x = raw(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    if not NET_INITTED:
        return x
    _pool('maxpool', input, x, kernel_size, stride, padding, ceil_mode)
    return x

def _adaptive_max_pool2d(raw, input, output_size, return_indices=False):
    x = raw(input, output_size, return_indices)
    if not NET_INITTED:
        return x
    if isinstance(output_size, int):
        out_dim = (output_size, output_size)
    else:
        out_dim = output_size
    tmp = (input.shape[2], input.shape[3])
    stride = (tmp[0] // out_dim[0], tmp[1] // out_dim[1])
    kernel_size = (tmp[0] - (out_dim[0] - 1) * stride[0], tmp[1] - (out_dim[1] - 1) * stride[1])
    _pool('maxpool', input, x, kernel_size, stride, 0, False)
    return x

def _avg_pool2d(raw, input, kernel_size, stride = None, padding = 0,
               ceil_mode = False, count_include_pad = True, divisor_override=None):
    x = raw(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    if not NET_INITTED:
        return x
    _pool('avgpool', input, x, kernel_size, stride, padding, ceil_mode)
    return x

def _adaptive_avg_pool2d(raw, input, output_size):
    x = raw(input, output_size)
    if not NET_INITTED:
        return x
    if isinstance(output_size, int):
        out_dim = (output_size, output_size)
    else:
        out_dim = output_size
    tmp = (input.shape[2], input.shape[3])
    stride = (tmp[0] // out_dim[0], tmp[1] // out_dim[1])
    kernel_size = (tmp[0] - (out_dim[0] - 1) * stride[0], tmp[1] - (out_dim[1] - 1) * stride[1])
    _pool('avgpool', input, x, kernel_size, stride, 0, False)
    return x

def _mean(input, dim, keepdim=False):
    x = raw_mean(input, dim, keepdim=keepdim)
    if not NET_INITTED:
        return x
    assert input.dim() == 4
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = dim
    # only support reduction along the width or height
    kernel_h, kernel_w = 1, 1
    for d in dims:
        if d == 2:
            kernel_h = input.shape[2]
        elif d == 3:
            kernel_w = input.shape[3]
        else:
            raise TypeError('unsupported reduction dim')

    if not keepdim:
        n, c, h, w = input.shape
        for d in dims:
            if d == 2:
                h = 1
            elif d == 3:
                w = 1
        tmp = torch.randn(n, c, h, w)
    else:
        tmp = x

    kernel_size = (kernel_h, kernel_w)
    stride = (1, 1)
    _pool('avgpool', input, tmp, kernel_size, stride, 0, False)

    if not keepdim:
        layer_name=log.add_layer(name='view')
        top_blobs=log.add_blobs([x],name='view_blob')
        layer = {}
        layer['name'] = layer_name
        layer['type'] = 'view'
        layer['bottom'] = log.blobs(tmp)
        layer['top'] = log.blobs(x)
        layer['bottom_shape'] = list(tmp.shape[1:])
        layer['top_shape'] = list(x.shape[1:])
        dims=list(x.shape)
        dims[0] = 0 # the first dim should be batch_size
        layer['new_shape'] = dims
        log.net.append(layer)


    return x


def _relu(raw, input, inplace=False):
    x = raw(input, inplace=False)
    if not NET_INITTED:
        return x
    name = log.add_layer(name='relu')
    top_blobs = log.add_blobs([x], name='relu_blob')
    layer = {}
    layer['name'] = name
    layer['type'] = 'relu'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    layer['activate'] = True
    log.net.append(layer)
    return x

def _leaky_relu(raw, input, negative_slope=0.1, inplace=False):
    x = raw(input, negative_slope)
    if not NET_INITTED:
        return x
    name = log.add_layer(name='leaky_relu')
    top_blobs = log.add_blobs([x], name='leaky_relu_blob')
    layer = {}
    layer['name'] = name
    layer['type'] = 'leaky_relu'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    layer['negative_slope'] = negative_slope
    hps = {}
    hps['negative_slope'] = negative_slope
    layer['hps'] = hps
    layer['activate'] = True
    log.net.append(layer)
    return x

def _batch_norm(raw,input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    x = raw(input, running_mean, running_var, weight, bias,
               training, momentum, eps)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='batchnorm')
    top_blobs = log.add_blobs([x], name='batch_norm_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'batchnorm'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    layer['running_mean'] = running_mean.data.cpu().numpy() 
    layer['running_var'] = running_var.data.cpu().numpy() 
    layer['eps'] = eps 
    layer['weight'] = weight.data.cpu().numpy() 
    layer['bias'] = bias.data.cpu().numpy() 
    layer['activate'] = True
    hps = {}
    hps['num_features'] = weight.shape[0]
    layer['hps'] = hps

    # check if there is a conv before
    for l in log.net:
        if l['top'] == layer['bottom']:
            if l['type'] != 'conv2d' and l['type'] != 'linear':
                has_conv_before = False
            else:
                has_conv_before = True
            break
    if not has_conv_before:
        conv_layer = {}
        conv_layer_name = log.add_layer(name='conv_bn')
        conv_layer['name'] = conv_layer_name
        conv_layer['type'] = 'conv2d'
        conv_layer['bottom'] = log.blobs(input)
        tmp_output = torch.randn(x.shape)
        top_blobs = log.add_blobs([tmp_output], name='conv_bn_blob')
        conv_layer['top'] = log.blobs(tmp_output)
        layer['bottom'] = log.blobs(tmp_output)
        conv_layer['bottom_shape'] = list(input.shape[1:])
        conv_layer['top_shape'] = list(tmp_output.shape[1:])
        weight = torch.ones(x.shape[1], 1, 1, 1)
        conv_layer['weight_name'] = conv_layer_name + '_weight'
        conv_layer['weight'] = weight.data.cpu().numpy()
        conv_hps = {}
        conv_hps['stride'] = (1, 1) 
        conv_hps['padding'] = (0, 0) 
        conv_hps['dilation'] = (1, 1) 
        conv_hps['groups'] = x.shape[1]
        conv_hps['out_channels'] = weight.shape[0]
        conv_hps['in_channels'] = weight.shape[1] * conv_hps['groups']
        conv_hps['kernel_size'] = (weight.shape[2], weight.shape[3])
        conv_hps['deconv_flag'] = 'nodeconv'
        conv_hps['deconv_height_factor'] = 1
        conv_hps['deconv_width_factor'] = 1
        conv_hps['height_factor'] = 1
        conv_hps['width_factor'] = 1
        conv_layer['bias'] = None
        conv_hps['bias_term'] = 0 #False
        conv_layer['hps'] = conv_hps
        log.net.append(conv_layer)
        
    log.net.append(layer)
    return x

def _cat(raw, inputs, dim=0):
    x = raw(inputs, dim=dim)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='concat')
    top_blobs = log.add_blobs([x], name='concat_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'concat'
    layer['bottom'] = []
    layer['bottom_shape'] = []
    for input_ in inputs:
        layer['bottom'].append(log.blobs(input_))
        layer['bottom_shape'].append(list(input_.shape[1:]))
    layer['top'] = log.blobs(x)
    layer['top_shape'] = list(x.shape[1:])
    layer['cat_dim'] = dim
    layer['cat_num'] = len(inputs)
    log.net.append(layer)
    return x

def _split(raw, input, split_size, dim=0):
    outputs = raw(input, split_size, dim)
    if not NET_INITTED:
        return outputs
    assert dim == 1
    layer_name = log.add_layer(name='split')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'split'
    layer['bottom'] = log.blobs(input)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top'] = []
    layer['top_shape'] = []
    for output_ in outputs:
        top_blob = log.add_blobs([output_], name='{}_blob'.format(layer_name))
        layer['top'].append(log.blobs(output_))
        layer['top_shape'].append(output_.shape[1:])
    layer['split_dim'] = dim
    layer['split_num'] = len(outputs)
    layer['split_size'] = split_size
    log.net.append(layer)
    return outputs


def _exp(raw, input):
    x = raw(input)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='exp')
    top_blobs = log.add_blobs([x], name='exp_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'exp'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)
    return x


def gen_bilinear_param(up_factor, in_channels):
    kernel_size = (up_factor[0] * 2 - up_factor[0] % 2, up_factor[1] * 2 - up_factor[1] % 2)
    if kernel_size[0] % 2 == 1:
        center_h = up_factor[0] - 1
    else:
        center_h = up_factor[0] - 0.5
    if kernel_size[1] % 2 == 1:
        center_w = up_factor[1] - 1
    else:
        center_w = up_factor[1] - 0.5
    og = np.ogrid[:kernel_size[0], :kernel_size[1]]
    param = (1 - abs(og[0] - center_h) / up_factor[0]) * ( 1 - abs(og[1] - center_w) / up_factor[1])
    param_all = np.zeros([int(in_channels), 1, int(kernel_size[0]), int(kernel_size[1])])
    for i in range(in_channels):
        param_all[i, 0, :, :] = param.copy()
    return param_all


def _interpolate(raw, input, size=None, scale_factor=None, mode='nearest', align_corners=False):
    if (mode != 'nearest' and mode != 'bilinear'): # or align_corners != False:
        raise NotImplementedError('currently only support nearest and bilinear without align_corners')
    if scale_factor == None:
        assert size[0] % input.shape[2] == 0
        assert size[1] % input.shape[3] == 0
        assert size[0] // input.shape[2] == size[1] // input.shape[3]
        height_factor = size[0] // input.shape[2]
        width_factor = size[1] // input.shape[3]
    else:
        height_factor = scale_factor[0] if isinstance(scale_factor, tuple) else int(scale_factor)
        width_factor = scale_factor[1] if isinstance(scale_factor, tuple) else int(scale_factor)
    x = raw(input, size, scale_factor, mode)
    if not NET_INITTED:
        return x
    if mode == 'nearest':
        remain_h, remain_w = height_factor, width_factor
        base_layer_name = log.add_layer(name='upsample')
        tmp_input = input
        while remain_h > 1 or remain_w > 1:
            h_factor, w_factor = 1, 1
            can_div_h, can_div_w = False, False
            for div_h in range(11, 1, -1):
                if remain_h % div_h == 0:
                    h_factor = div_h
                    can_div_h = True
                    break
            for div_w in range(11, 1, -1):
                if remain_w % div_w == 0:
                    w_factor = div_w
                    can_div_w = True
                    break
            assert can_div_h == True and can_div_w == True
            remain_h = remain_h // h_factor
            remain_w = remain_w // w_factor
            if remain_h == 1 and remain_w == 1:
                tmp_output = x
            else:
                tmp_output = torch.randn(1, tmp_input.shape[1], tmp_input.shape[2] * h_factor, tmp_input.shape[3] * w_factor)

            layer_name = log.add_layer(name=base_layer_name + '_level')
            top_blobs = log.add_blobs([tmp_output], name=layer_name[:-1] + '_blob')
            layer = {}
            layer['name'] = layer_name
            layer['type'] = 'upsample'
            layer['bottom'] = log.blobs(tmp_input)
            layer['top'] = log.blobs(tmp_output)
            layer['bottom_shape'] = list(tmp_input.shape[1:])
            layer['top_shape'] = list(tmp_output.shape[1:])
            layer['height_factor'] = h_factor
            layer['width_factor'] = w_factor
            log.net.append(layer)
            tmp_input = tmp_output

    elif mode == 'bilinear':
        layer_name = log.add_layer(name='upsample_deconv')
        top_blobs = log.add_blobs([x], name='upsample_deconv_blob')
        layer = {}
        layer['name'] = layer_name
        layer['type'] = 'deconv'
        layer['bottom'] = log.blobs(input)
        layer['top'] = log.blobs(x)
        layer['bottom_shape'] = list(input.shape[1:])
        layer['top_shape'] = list(x.shape[1:])
        weight = gen_bilinear_param((height_factor, width_factor), x.shape[1])
        layer['weight'] = weight 
        hps = {}
        hps['stride'] = (int(height_factor), int(width_factor)) 
        hps['padding'] = (int(height_factor) // 2, int(width_factor) // 2) 
        hps['dilation'] = (1, 1) 
        hps['groups'] = x.shape[1]
        hps['out_channels'] = weight.shape[0]
        hps['in_channels'] = weight.shape[1] * hps['groups']
        hps['kernel_size'] = (weight.shape[2], weight.shape[3])
        layer['bias'] = None
        hps['bias_term'] = 0
        layer['hps'] = hps
        log.net.append(layer)
    
    return x

def _hardtanh(raw, input, min_val, max_val, inplace):
    x = raw(input, min_val, max_val, inplace=False)
    if not NET_INITTED:
        return x
    '''
    relu_name = log.add_layer(name='relu6_relu')
    relu_layer = {}
    relu_layer['name'] = relu_name
    relu_layer['type'] = 'relu'
    relu_layer['bottom'] = log.blobs(input)
    tmp = torch.randn(x.shape)
    top_blobs = log.add_blobs([tmp], name='relu6_relu_blob')
    relu_layer['top'] = log.blobs(tmp)
    relu_layer['bottom_shape'] = list(input.shape[1:])
    relu_layer['top_shape'] = list(tmp.shape[1:])
    relu_layer['activate'] = True
    relu_layer['sub_type'] = 'relu6'
    log.net.append(relu_layer)
    '''

    #'''
    layer_name = log.add_layer(name='relu6')
    top_blobs = log.add_blobs([x], name='relu6_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'relu6'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    layer['activate'] = True
    log.net.append(layer)
    #'''
    return x

def _sigmoid(raw, input):
    x = raw(input)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='sigmoid')
    top_blobs = log.add_blobs([x], name='sigmoid_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'sigmoid'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)
    return x


# not verified, unfinished
def _softmax(raw, input, dim=None, _stacklevel=3):
    x = raw(input, dim=dim, _stacklevel=_stacklevel)
    if not NET_INITTED:
        return x
    #only support two scenarios: NxC and NxCxHxW. DO NOT permute the dims.
    assert dim == 1
    #transform to exp -> sum(conv) -> inverse -> expand -> multiply
    #exp
    layer_name = log.add_layer(name='softmax_exp')
    exp_temp = torch.randn(input.shape)
    top_blobs = log.add_blobs([exp_temp], name='softmax_exp_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'exp'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(exp_temp)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)

    #sum and expand
    sum_type = 'conv2d' if exp_temp.dim() == 4 else 'linear'
    if sum_type == 'conv2d':
        layer_name = log.add_layer(name='softmax_conv')
        sum_temp = torch.randn(input.shape)
        top_blobs = log.add_blobs([sum_temp], name='softmax_conv_blob')
        layer = {}
        layer['name'] = layer_name
        layer['type'] = 'conv2d'
        layer['bottom'] = log.blobs(exp_temp)
        layer['bottom_shape'] = list(exp_temp.shape[1:])
        layer['top'] = log.blobs(sum_temp)
        layer['top_shape'] = list(sum_temp.shape[1:])
        weight = torch.ones(input.shape[1], input.shape[1], 1, 1)
        layer['weight'] = weight.data.cpu().numpy()
        hps = {}
        hps['stride'] = (1, 1)
        hps['padding'] = (0, 0)
        hps['dilation'] = (1, 1)
        hps['groups'] = 1
        hps['out_channels'] = weight.shape[0]
        hps['in_channels'] = weight.shape[1] * hps['groups']
        hps['kernel_size'] = (weight.shape[2], weight.shape[3])
        hps['deconv_flag'] = 'nodeconv'
        hps['deconv_height_factor'] = 1
        hps['deconv_width_factor'] = 1
        hps['height_factor'] = 1
        hps['width_factor'] = 1
        layer['bias'] = None
        hps['bias_term'] = 0
        layer['hps'] = hps
        log.net.append(layer)

    elif sum_type == 'linear':
        layer_name=log.add_layer(name='softmax_fc')
        sum_temp = torch.randn(input.shape)
        top_blobs=log.add_blobs([sum_temp],name='softmax_fc_blob')
        layer = {}
        layer['name'] = layer_name
        layer['type'] = 'linear'
        layer['bottom'] = log.blobs(exp_temp)
        layer['bottom_shape'] = list(exp_temp.shape[1:])
        layer['top'] = log.blobs(sum_temp)
        layer['top_shape'] = list(sum_temp.shape[1:])
        weight = torch.ones(input.shape[1], input.shape[1]) 
        layer['weight'] = weight.data.cpu().numpy() 
        hps = {}
        hps['out_features'] = weight.shape[0]
        hps['in_features'] = weight.shape[1]
        hps['height_factor'] = 1
        hps['width_factor'] = 1
        layer['bias'] = None
        hps['bias_term'] = 0 #False
        layer['hps'] = hps
        log.net.append(layer)

    #inverse
    layer_name = log.add_layer(name='softmax_inv')                    
    inv_temp = torch.randn(sum_temp.shape)
    top_blobs = log.add_blobs([inv_temp], name='softmax_inv_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'inv'
    layer['bottom'] = log.blobs(sum_temp)
    layer['top'] = log.blobs(inv_temp)
    layer['bottom_shape'] = list(sum_temp.shape[1:])
    layer['top_shape'] = list(inv_temp.shape[1:])
    log.net.append(layer)

    #multiply
    layer_name = log.add_layer(name='softmax_mul')
    top_blobs = log.add_blobs([x], name='softmax_mul_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'mul'
    layer['bottom'] = [log.blobs(exp_temp), log.blobs(inv_temp)]
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = [list(exp_temp.shape[1:]), list(inv_temp.shape[1:])]
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)

    return x

# ----- for Variable operations --------

def _pow(input, exponent):
    assert isinstance(exponent, float) or isinstance(exponent, int)
    x = raw_pow(input, exponent)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='pow')
    top_blobs = log.add_blobs([x], name='pow_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'pow'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    hps = {}
    hps['exponent'] = exponent
    layer['hps'] = hps
    log.net.append(layer)
    return x


def _expand_as(input, *args):
    #support (N, 1, 1, 1) -> (N, C, H, W), (N, 1, H, W) -> (N, C, H, W), (N, C, 1, 1) -> (N, C, H, W)

    x = raw_expand_as(input, *args)
    if not NET_INITTED:
        return x

    if input.shape[1] == 1 and x.shape[1] > 1:
        weight = torch.ones(x.shape[1], 1, 1, 1)
        name = log.add_layer(name='expand_conv')
        layer = {}
        layer['name'] = name
        layer['type'] = 'conv2d'
        layer['bottom'] = log.blobs(input)
        layer['bottom_shape'] = list(input.shape[1:])
        if (input.shape[2] == 1 and x.shape[2] > 1) or (input.shape[3] == 1 and x.shape[3] > 1):
            tmp = torch.randn(x.shape[0], x.shape[1], input.shape[2], input.shape[3])
            layer['top'] = log.blobs(tmp)
            layer['top_shape'] = list(tmp.shape[1:])
            log.add_blobs([tmp], name='expand_conv_blob')
        else:
            layer['top'] = log.blobs(x)
            layer['top_shape'] = list(x.shape[1:])
            log.add_blobs([x], name='expand_conv_blob')
        layer['weight'] = weight.data.cpu().numpy()
        hps = {}
        hps['stride'] = (1, 1) 
        hps['padding'] = (0, 0) 
        hps['dilation'] = (1, 1) 
        hps['groups'] = 1 #groups
        hps['out_channels'] = weight.shape[0]
        hps['in_channels'] = weight.shape[1] * hps['groups']
        hps['kernel_size'] = (weight.shape[2], weight.shape[3])
        hps['deconv_flag'] = 'nodeconv'
        hps['deconv_height_factor'] = 1
        hps['deconv_width_factor'] = 1
        hps['height_factor'] = 1
        hps['width_factor'] = 1
        layer['bias'] = None
        hps['bias_term'] = 0 #False
        layer['hps'] = hps
        log.net.append(layer)

    if (input.shape[2] == 1 and x.shape[2] > 1) or (input.shape[3] == 1 and x.shape[3] > 1):
        if input.shape[1] == 1 and x.shape[1] > 1:
            tmp_input = tmp
        else:
            tmp_input = input
        height_factor = x.shape[2] // input.shape[2]
        width_factor = x.shape[3] // input.shape[3]
        remain_h, remain_w = height_factor, width_factor
        base_layer_name = log.add_layer(name='upsample')
        while remain_h > 1 or remain_w > 1:
            h_factor, w_factor = 1, 1
            can_div_h, can_div_w = False, False
            for div_h in range(11, 1, -1):
                if remain_h % div_h == 0:
                    h_factor = div_h
                    can_div_h = True
                    break
            for div_w in range(11, 1, -1):
                if remain_w % div_w == 0:
                    w_factor = div_w
                    can_div_w = True
                    break
            assert can_div_h == True and can_div_w == True
            remain_h = remain_h // h_factor
            remain_w = remain_w // w_factor
            if remain_h == 1 and remain_w == 1:
                tmp_output = x
            else:
                tmp_output = torch.randn(1, tmp_input.shape[1], tmp_input.shape[2] * h_factor, tmp_input.shape[3] * w_factor)
                                                                                                                              
            layer_name = log.add_layer(name=base_layer_name + '_level')
            top_blobs = log.add_blobs([tmp_output], name=layer_name[:-1] + '_blob')
            layer = {}
            layer['name'] = layer_name
            layer['type'] = 'upsample'
            layer['bottom'] = log.blobs(tmp_input)
            layer['top'] = log.blobs(tmp_output)
            layer['bottom_shape'] = list(tmp_input.shape[1:])
            layer['top_shape'] = list(tmp_output.shape[1:])
            layer['height_factor'] = h_factor
            layer['width_factor'] = w_factor
            log.net.append(layer)
            tmp_input = tmp_output
    return x

def _flatten(raw, input, start_dim=0, end_dim=-1):
    x = raw(input, start_dim, end_dim)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='view')
    top_blobs = log.add_blobs([x], name='view_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'view'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    if end_dim == -1:
        end_dim = len(x.shape) - 1
    dims = list(x.shape[0:start_dim])
    dims.append(1)
    for i in range(start_dim, end_dim + 1):
        dims[-1] *= x.shape[i]
    dims += list(x.shape[end_dim+1:])
    dims[0] = 0
    layer['new_shape'] = dims
    log.net.append(layer)
    return x

def _view(input, *args):
    x=raw_view(input, *args)
    if not NET_INITTED:
        return x
    layer_name=log.add_layer(name='view')
    top_blobs=log.add_blobs([x],name='view_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'view'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    dims=list(args)
    dims[0] = 0 # the first dim should be batch_size
    layer['new_shape'] = dims
    log.net.append(layer)
    return x

def _add(input, *args):
    x = raw__add__(input, *args)
    global NET_INITTED
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'add'
    if log.blobs(args[0], allow_unfound=True) is None:
        layer['const'] = True
        NET_INITTED = False
        if isinstance(args[0], float) or isinstance(args[0], int):
            const_value = torch.tensor([args[0]]).expand_as(input)
        else:
            const_value = args[0].expand_as(input)
        NET_INITTED = True
        layer['const_value'] = const_value.data.cpu().numpy()
        layer['bottom'] = [log.blobs(input), layer_name + '_const_blob']
    else:
        layer['const'] = False
        layer['bottom'] = [log.blobs(input), log.blobs(args[0])]

    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = [list(input.shape[1:]), list(input.shape[1:])]
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)
    return x

def _iadd(input, *args):
    x = raw__iadd__(input, *args)
    global NET_INITTED
    if not NET_INITTED:
        return x
    x=x.clone() # to avoid in-place operation
    layer_name = log.add_layer(name='add')
    top_blobs = log.add_blobs([x], name='add_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'add'
    if log.blobs(args[0], allow_unfound=True) is None:
        layer['const'] = True
        NET_INITTED = False
        if isinstance(args[0], float) or isinstance(args[0], int):
            const_value = torch.tensor([args[0]]).expand_as(input)
        else:
            const_value = args[0].expand_as(input)
        NET_INITTED = True
        layer['const_value'] = const_value.data.cpu().numpy()
        layer['bottom'] = [log.blobs(input), layer_name + '_const_blob']
    else:
        layer['const'] = False
        layer['bottom'] = [log.blobs(input), log.blobs(args[0])]

    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = [list(input.shape[1:]), list(input.shape[1:])]
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)
    return x

def _mul(input, *args):
    x = raw__mul__(input, *args)
    global NET_INITTED
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'mul'
    if log.blobs(args[0], allow_unfound=True) is None:
        layer['const'] = True
        NET_INITTED = False
        if isinstance(args[0], float) or isinstance(args[0], int):
            const_value = torch.tensor([args[0]]).expand_as(input)
        else:
            const_value = args[0].expand_as(input)
        NET_INITTED = True
        layer['const_value'] = const_value.data.cpu().numpy()
        layer['bottom'] = [log.blobs(input), layer_name + '_const_blob']
    else:
        layer['const'] = False
        layer['bottom'] = [log.blobs(input), log.blobs(args[0])]
    
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = [list(input.shape[1:]), list(input.shape[1:])]
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)
    return x

def _imul(input, *args):
    x = raw__imul__(input, *args)
    global NET_INITTED
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='mul')
    top_blobs = log.add_blobs([x], name='mul_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'mul'
    if log.blobs(args[0], allow_unfound=True) is None:                
        layer['const'] = True
        NET_INITTED = False
        if isinstance(args[0], float) or isinstance(args[0], int):
            const_value = torch.tensor([args[0]]).expand_as(input)
        else:
            const_value = args[0].expand_as(input)
        NET_INITTED = True
        layer['const_value'] = const_value.data.cpu().numpy()
        layer['bottom'] = [log.blobs(input), layer_name + '_const_blob']
    else:
        layer['const'] = False
        layer['bottom'] = [log.blobs(input), log.blobs(args[0])]

    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = [list(input.shape[1:]), list(input.shape[1:])]
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)
    return x

def _permute(input, *args):
    x = raw_permute(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='permute')
    top_blobs = log.add_blobs([x], name='permute_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'permute'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    dims = list(args)
    #dims[0] = 0 # the first dim should be batch_size
    layer['permute_dim'] = dims
    log.net.append(layer)
    return x

def _contiguous(input, *args):
    x = raw_contiguous(input, *args)
    if not NET_INITTED:
        return x
    layer_name = log.add_layer(name='contiguous')
    top_blobs = log.add_blobs([x], name='contiguous_blob')
    layer = {}
    layer['name'] = layer_name
    layer['type'] = 'contiguous'
    layer['bottom'] = log.blobs(input)
    layer['top'] = log.blobs(x)
    layer['bottom_shape'] = list(input.shape[1:])
    layer['top_shape'] = list(x.shape[1:])
    log.net.append(layer)
    return x


class Rp(object):
    def __init__(self,raw,replace,**kwargs):
        # replace the raw function to replace function
        self.obj=replace
        self.raw=raw

    def __call__(self,*args,**kwargs):
        if not NET_INITTED:
            return self.raw(*args,**kwargs)
        out=self.obj(self.raw,*args,**kwargs)
        return out

F.conv2d=Rp(F.conv2d,_conv2d) 
F.conv_transpose2d=Rp(F.conv_transpose2d, _conv_transpose2d)
F.linear=Rp(F.linear,_linear) 
F.relu=Rp(F.relu,_relu) 
F.leaky_relu = Rp(F.leaky_relu, _leaky_relu)
F.max_pool2d=Rp(F.max_pool2d,_max_pool2d) 
F.avg_pool2d=Rp(F.avg_pool2d,_avg_pool2d)
F.adaptive_avg_pool2d = Rp(F.adaptive_avg_pool2d, _adaptive_avg_pool2d) 
F.adaptive_max_pool2d = Rp(F.adaptive_max_pool2d, _adaptive_max_pool2d)
F.batch_norm=Rp(F.batch_norm,_batch_norm) 
F.interpolate = Rp(F.interpolate, _interpolate)
F.hardtanh = Rp(F.hardtanh, _hardtanh)
F.sigmoid = Rp(F.sigmoid, _sigmoid)
F.softmax = Rp(F.softmax, _softmax)

torch.cat = Rp(torch.cat, _cat)
torch.flatten = Rp(torch.flatten, _flatten)
torch.sigmoid = Rp(torch.sigmoid, _sigmoid)
torch.exp = Rp(torch.exp, _exp)
torch.split = Rp(torch.split, _split)

try:
    raw_view=Variable.view 
    Variable.view=_view 
    raw__add__=Variable.__add__ 
    Variable.__add__=_add 
    raw__iadd__=Variable.__iadd__ 
    Variable.__iadd__=_iadd
    raw__mul__ = Variable.__mul__
    Variable.__mul__ = _mul
    raw__imul__ = Variable.__imul__
    Variable.__imul__ = _imul
except:
    # for new version 0.4.0 and later version
    for t in [torch.Tensor]:
        raw_view = t.view 
        t.view = _view 
        raw__add__ = t.__add__ 
        t.__add__ = _add 
        raw__iadd__ = t.__iadd__ 
        t.__iadd__ = _iadd
        raw__mul__ = t.__mul__
        t.__mul__ = _mul
        raw__imul__ = t.__imul__
        t.__imul__ = _imul
        raw_permute = t.permute
        t.permute = _permute
        raw_contiguous = t.contiguous
        t.contiguous = _contiguous
        raw_expand_as = t.expand_as
        t.expand_as = _expand_as
        raw_mean = t.mean
        t.mean = _mean
        raw_pow = t.pow
        t.pow = _pow

def extract_arch(model, input_var):
    model.eval()
    log.init([input_var])
    global NET_INITTED
    NET_INITTED = True
    out = model.forward(input_var)
    NET_INITTED = False
    return_blobs = []
    if isinstance(out, list) or isinstance(out, tuple):
        for out_blob in out:
            return_blobs.append(log.blobs(out_blob))
    else:
        return_blobs.append(log.blobs(out))
    net = log.net
    log.reset()
    return net, return_blobs
