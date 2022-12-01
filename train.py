import argparse
import os

from models import ImagenetRunConfig
from rgn_manager import *
from ensemble_in_one import EnsembleInOneNets
from advertorch.utils import NormalizeByChannelMeanStd
from resnet import resnet
from vgg import vgg16

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='exp')
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--pretrained_model', default='', type=str)
parser.add_argument('--arch', default='resnet', type=str)

""" run config """
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--warmup_epochs', type=int, default=200)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--init_lr', type=float, default=0.1)
parser.add_argument('--lr_schedule_type', type=str, default='step')
parser.add_argument('--lr_schedule_param', type=list, default=[100, 150])

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--valid_size', type=int, default=10000)

parser.add_argument('--optim_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--label_smoothing', type=float, default=0)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=2)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='normal', choices=['normal', 'strong', 'None'])

""" Distill hyper-prameters """
parser.add_argument('--distill_eps', type=float, default=0.07)
parser.add_argument('--distill_alpha', type=float, default=0.007)
parser.add_argument('--distill_steps', type=int, default=10)
parser.add_argument('--distill_batch', type=int, default=3)
parser.add_argument('--distill_fix', action='store_true')
parser.add_argument('--distill_layer', type=int, default=0)
parser.add_argument('--distill_train_mode', action='store_true')
parser.add_argument('--layers', type=int, default=20)
parser.add_argument('--limit_max_layer', type=int, default=0)
parser.add_argument('--aug_num', type=int, default=2)
parser.add_argument('--adv_train', action='store_true')


class model_wrapper(nn.Module):
    def __init__(self, model, normalizer=None):
        super(model_wrapper, self).__init__()
        self.model = model
        self.normalizer = normalizer

    def forward(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return self.model(x)

    def get_features(self, x, layer, before_relu=False):
        if self.normalizer:
            x = self.normalizer(x)
        x = self.model.get_features(x, layer, before_relu)
        return x

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.optim_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = ImagenetRunConfig(
        **args.__dict__
    )

    # debug, adjust run_config
    if args.debug:
        run_config.train_batch_size = 256
        run_config.test_batch_size = 256
        run_config.valid_size = 256
        run_config.n_worker = 0

    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).cuda()
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()
    normalizer = NormalizeByChannelMeanStd(mean=mean, std=std)
   
    if args.arch == 'resnet':
        model = resnet(depth=20, num_classes=10, aug_num=args.aug_num)
    elif args.arch == 'vgg':
        model = vgg16()
    model = model_wrapper(model, normalizer)
    super_net = EnsembleInOneNets(model)
    print('Model Construction:')
    print(super_net)

    # build arch search config from args

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # arch search run manager
    rgn_run_manager = EnsembleInOneRunManager(args.path, super_net, run_config)
    if args.warmup:
        rgn_run_manager.warmup = True

    # resume
    if args.resume:
        try:
            rgn_run_manager.load_model()
        except Exception:
            from pathlib import Path
            home = str(Path.home())
            warmup_path = 'exp_warmup/checkpoint/warmup.pth.tar'
            if os.path.exists(warmup_path):
                print('load warmup weights')
                #arch_search_run_manager.load_model(model_fname=warmup_path)
                checkpoint = torch.load(warmup_path)
                super_net.module.load_state_dict(checkpoint)
            else:
                print('fail to load models')

    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model)['state_dict']
        super_net.load_state_dict(checkpoint, strict=False)


    # warmup
    if rgn_run_manager.warmup:
        rgn_run_manager.warm_up(args.warmup_epochs)

    # training with distilled data
    rgn_run_manager.run_manager.init_lr = args.init_lr
    rgn_run_manager.train(args)
