import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def stem(op, has_bn=True):
    if has_bn:
        stem = nn.Sequential(op, nn.BatchNorm2d(op.out_channels))
    else:
        stem = op
    return stem

def build_candidate_ops(op, has_bn=True, n_path=2):
    if op is None:
        raise ValueError('candidate op cannot be None')
    
    return [
        stem(op, has_bn) for i in range(n_path)
    ]

class RGB(nn.Module):
    def __init__(self, op, has_bn=True, npath=2):
        super(RGB, self).__init__()
        assert isinstance(op, nn.Conv2d)
        candidate_ops = build_candidate_ops(op, has_bn, npath)
        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.zeros(self.n_choices)) #Tensor -> zeros
        self.AP_path_wb = Parameter(torch.zeros(self.n_choices)) #Tensor -> zeros

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    @property
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.candidate_ops[index]

    @property
    def active_op(self):
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def forward(self, x):
        output = self.active_op(x)
        return output

    def binarize(self, random=False):
        self.log_prob = None
        self.AP_path_wb.data.zero_()
        if random:
            self.AP_path_alpha.data.zero_()# = torch.zeros(self.AP_path_alpha.shape)#.zero_()
        probs = self.probs_over_ops

        sample = torch.multinomial(probs.data, 1)[0].item()
        self.active_index = [sample]
        self.inactive_index = [_i for _i in range(0, sample)] + \
                              [_i for _i in range(sample+1, self.n_choices)]
        self.log_prob = torch.log(probs[sample])
        self.current_prob_over_ops = probs
        self.AP_path_wb.data[sample] = 1.0
        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None


if __name__ == '__main__':
    l = RGB(nn.Conv2d(10, 10, kernel_size=3), has_bn=True, npath=3)
    import torch
    print(l)
    l.binarize()
    print(l.active_op)
    a = torch.randn(1, 10, 20, 20)
    print(l(a).shape)
