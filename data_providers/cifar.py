# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *

from tqdm import tqdm

class DistillationLoader:
    def __init__(self, seed, target):
        self.seed = iter(seed)
        self.target = iter(target)

    def __len__(self):
        return len(self.seed)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            si, sl = next(self.seed)
            ti, tl = next(self.target)
            return si, sl, ti, tl
        except StopIteration as e:
            raise StopIteration

class CifarDataProvider(DataProvider):

    def __init__(self, save_path='./data/', train_batch_size=128, test_batch_size=500, valid_size=None, n_worker=4, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        train_transforms, val_transforms = self.build_train_transform()
        train_dataset = datasets.CIFAR10(
            root=self._save_path, train=True, transform=train_transforms, download=True
        )

        valid_dataset = datasets.CIFAR10(
            root=self._save_path, train=False, transform=val_transforms, download=True
        )

        self.train = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, sampler=None, shuffle=True,
            num_workers=n_worker, pin_memory=True,
        )
        self.valid = torch.utils.data.DataLoader(
            valid_dataset, batch_size=test_batch_size, sampler=None, shuffle=False,
            num_workers=n_worker, pin_memory=True,
        )

        '''
        train_dataset2 = datasets.CIFAR10(
            root=self._save_path, train=True, transform=train_transforms, download=True
        )

        self.train2 = torch.utils.data.DataLoader(
            train_dataset2, batch_size=train_batch_size, sampler=None, shuffle=True,
            num_workers=n_worker, pin_memory=True,
        )
        '''

        #self.distill_loader = DistillationLoader(self.train, self.train)

        self.test = self.valid


    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = './data/'
        return self._save_path


    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    def build_train_transform(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            #self.normalize
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            #self.normalize
        ])
        return train_transform, val_transform

    @property
    def image_size(self):
        return 32
