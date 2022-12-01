import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def linf_distillation(model, dat, target, distill_layer=None, eps=0.05, alpha=0.005, steps=10, rand_start=True, train_mode=False, \
        criterion=nn.MSELoss()):
    if train_mode:
        model.train()
    else:
        model.eval()
    x_nat = dat.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = dat.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)
    for i in range(steps):
        x_adv.requires_grad = True
        if not distill_layer:
            outputs = model(x_adv)
            targets = model(target).data.clone().detach()
        else:
            outputs = model.get_features(x_adv, distill_layer)
            targets = model.get_features(target, distill_layer).data.clone().detach()
        loss = criterion(outputs, targets)
        model.zero_grad()
        loss.backward()
        data_grad = x_adv.grad.data

        with torch.no_grad():
            x_adv = x_adv - alpha * data_grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            x_adv = torch.clamp(x_adv, 0., 1.)
    model.train()
    return x_adv.clone().detach()
