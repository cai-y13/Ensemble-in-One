import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def gen_adv_FGSM(model, criterion, x, y, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
    x_adv = x.clone()
    x_adv.requires_grad = True
    model.eval()
    h_adv = model(x_adv)
    if targeted:
        loss = criterion(h_adv, y)
    else:
        loss = -criterion(h_adv, y)
    model.zero_grad()
    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    loss.backward()
    x_adv.grad.sign_()
    x_adv = x_adv - x_adv.grad * eps

    return x_adv

def gen_adv_iFGSM(model, criterion, x, y, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1, rand_start=True):
    x_nat = x.clone().detach()
    x_adv = None
    if rand_start:
        x_adv = x.clone().detach() + torch.FloatTensor(x.shape).uniform_(-eps, eps).cuda()
    else:
        x_adv = x.clone().detach()
    x_adv = torch.clamp(x_adv, 0., 1.)

    for i in range(iteration):
        x_adv.requires_grad = True
        h_adv = model(x_adv)
        if targeted:
            loss = criterion(h_adv, y)
        else:
            loss = -criterion(h_adv, y)
        model.zero_grad()
        loss.backward()
        data_grad = x_adv.grad.data

        with torch.no_grad(): 
            x_adv = x_adv - data_grad.sign() * alpha
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            x_adv = torch.clamp(x_adv, 0., 1.)

    return x_adv.clone().detach()

def gen_adv_PGD(model, criterion, x, y, targeted=False, eps=0.03, alpha=1, iteration=1, random_init=True, x_val_min=-1, x_val_max=1):
    x_adv = x.clone()
    model.eval()
    if random_init:
        x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * eps

    for i in range(iteration):
        x_adv.requires_grad = True
        h_adv = model(x_adv)
        if targeted:
            loss = criterion(h_adv, y)
        else:
            loss = -criterion(h_adv, y)
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward()
        x_adv.grad.sign_()
        x_adv = x_adv - x_adv.grad * alpha
        x_adv = torch.where(x_adv > x+eps, x+eps, x_adv)
        x_adv = torch.where(x_adv < x-eps, x-eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

    return x_adv


