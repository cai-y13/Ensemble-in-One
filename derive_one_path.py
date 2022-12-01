import torch
import torch.nn as nn

from derive_model.extract_model_arch import extract_arch
from derive_model.derive import derive_model, derive_forward, derive_checkpoint
from derive_model.transform_tools import construct_model, construct_weight

from resnet import resnet
from vgg import vgg16
from ensemble_in_one import EnsembleInOneNets
from train import model_wrapper
from advertorch.utils import NormalizeByChannelMeanStd

mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
model = model_wrapper(resnet(depth=20, num_classes=10), normalizer)
super_net = EnsembleInOneNets(model)


checkpoint = torch.load('exp_resnet20_aug2_batch3_eps007/checkpoint/model.pth.tar')['state_dict']
super_net.load_state_dict(checkpoint, strict=False)

super_net.random_reset_binary_gates()

m = super_net.net.model

fake_input = torch.randn(1, 3, 32, 32)

arch, return_blobs = extract_arch(m, fake_input)
construct_model(arch, return_blobs, 'tmp_file')
construct_weight(arch, 'tmp_file')


