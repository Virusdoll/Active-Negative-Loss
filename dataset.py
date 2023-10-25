from torchvision import datasets, transforms
from torch.utils.data import random_split
from PIL import Image
import torch
import numpy as np
import os

MNIST_MEAN = [0.1307]
MNIST_STD = [0.3081]
CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]
CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_sym_T(eta, num_classes):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    diag_mask = np.eye(num_classes)
    rest_mask = 1 - diag_mask
    
    T = diag_mask * (1 - eta) \
        + rest_mask * eta / (num_classes - 1)
    
    return T

def get_asym_T_mnist(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # 7 -> 1
    T[7, 7], T[7, 1] = 1. - eta, eta
    # 2 -> 7
    T[2, 2], T[2, 7] = 1. - eta, eta
    # 5 <-> 6
    T[5, 5], T[5, 6] = 1. - eta, eta
    T[6, 6], T[6, 5] = 1. - eta, eta
    # 3 -> 8
    T[3, 3], T[3, 8] = 1. - eta, eta
    
    return T

def get_asym_T_cifar10(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 10
    
    T = np.eye(num_classes)
    # truck -> automobile (9 -> 1)
    T[9, 9], T[9, 1] = 1. - eta, eta
    # bird -> airplane (2 -> 0)
    T[2, 2], T[2, 0] = 1. - eta, eta
    # cat <-> dog (3 <-> 5)
    T[3, 3], T[3, 5] = 1. - eta, eta
    T[5, 5], T[5, 3] = 1. - eta, eta
    # deer -> horse (4 -> 7)
    T[4, 4], T[4, 7] = 1. - eta, eta
    
    return T
    
def get_asym_T_cifar100(eta):
    '''
    eta: noise rate
    '''
    assert (eta >= 0.) and (eta <= 1.)
    
    num_classes = 100
    num_superclasses = 20
    num_subclasses = 5

    T = np.eye(num_classes)

    for i in np.arange(num_superclasses):
        # build T for one superclass
        T_superclass = (1. - eta) * np.eye(num_subclasses)
        for j in np.arange(num_subclasses - 1):
            T_superclass[j, j + 1] = eta
        T_superclass[num_subclasses - 1, 0] = eta
        
        init, end = i * num_subclasses, (i + 1) * num_subclasses
        T[init:end, init:end] = T_superclass

    return T

def create_noisy_labels(labels, trans_matrix):
    '''
    create noisy labels from labels and noisy matrix
    '''
    
    if trans_matrix is None:
        raise ValueError('Noisy matrix is None')
    
    num_trans_matrix = trans_matrix.copy()
    labels = labels.copy()
    
    num_classes = len(trans_matrix)
    class_idx = [np.where(np.array(labels) == i)[0]
                 for i in range(num_classes)]
    num_samples_class = [len(class_idx[idx])
                         for idx in range(num_classes)]
    for real_label in range(num_classes):
        for trans_label in range(num_classes):
            num_trans_matrix[real_label][trans_label] = \
                trans_matrix[real_label][trans_label] * num_samples_class[real_label]
    num_trans_matrix = num_trans_matrix.astype(int)

    for real_label in range(num_classes):
        for trans_label in range(num_classes):

            if real_label == trans_label:
                continue

            num_trans = num_trans_matrix[real_label][trans_label]
            if num_trans == 0:
                continue

            trans_samples_idx = np.random.choice(class_idx[real_label],
                                                 num_trans,
                                                 replace=False)
            class_idx[real_label] = np.setdiff1d(class_idx[real_label],
                                                 trans_samples_idx)
            for idx in trans_samples_idx:
                labels[idx] = trans_label
    
    return labels

class MyMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets.numpy(), trans_matrix)

class MyCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets, trans_matrix)

class MyCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, trans_matrix=None):
        super().__init__(root, train, transform, target_transform, download)
        
        self.trans_matrix = trans_matrix
        if self.trans_matrix is not None:
            self.targets = create_noisy_labels(self.targets, trans_matrix)

class WebVisionDataset:
    def __init__(self, root, file_name='webvision_mini_train.txt',
                 transform=None, target_transform=None):
        self.target_list = []
        self.root = root
        self.load_file(os.path.join(root, file_name))
        self.transform = transform
        self.target_transform = target_transform
        return

    def load_file(self, filename):
        f = open(filename, "r")
        for line in f:
            train_file, label = line.split()
            self.target_list.append((train_file, int(label)))
        f.close()
        return

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        impath, target = self.target_list[index]
        img = Image.open(os.path.join(self.root, impath)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class ImageNetMini(datasets.ImageNet):
    def __init__(self, root, split='val', **kwargs):
        super(ImageNetMini, self).__init__(root, split=split, **kwargs)
        self.new_targets = []
        self.new_images = []
        for i, (file, cls_id) in enumerate(self.imgs):
            if cls_id <= 49:
                self.new_targets.append(cls_id)
                self.new_images.append((file, cls_id))
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs
        return

def mnist(root, noise_type, noise_rate, tuning=False):
    if noise_type == 'sym':
        T = get_sym_T(noise_rate, 10)
    elif noise_type == 'asym':
        T = get_asym_T_mnist(noise_rate)
    else:
        raise ValueError('Wrong noise type! Must be sym or asym')
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)])
    
    if tuning:
        train_dataset = MyMNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        num_train = int(len(train_dataset) * 0.9)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T
        
        eval_dataset = MyMNIST(root=root,
                               train=True,
                               transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:
        train_dataset = MyMNIST(root=root,
                                train=True,
                                transform=train_transform,
                                trans_matrix=T)
        
        eval_dataset = MyMNIST(root=root,
                               train=False,
                               transform=eval_transform)
    
    return train_dataset, eval_dataset

def cifar10(root, noise_type, noise_rate, tuning=False):
    if noise_type == 'sym':
        T = get_sym_T(noise_rate, 10)
    elif noise_type == 'asym':
        T = get_asym_T_cifar10(noise_rate)
    else:
        raise ValueError('Wrong noise type! Must be sym or asym')
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    
    if tuning:
        train_dataset = MyCIFAR10(root=root,
                                  train=True,
                                  transform=train_transform,
                                  trans_matrix=T)
        num_train = int(len(train_dataset) * 0.9)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T

        eval_dataset = MyCIFAR10(root=root,
                                 train=True,
                                 transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:
        train_dataset = MyCIFAR10(root=root,
                                  train=True,
                                  transform=train_transform,
                                  trans_matrix=T)

        eval_dataset = MyCIFAR10(root=root,
                                 train=False,
                                 transform=eval_transform)
    
    return train_dataset, eval_dataset

def cifar100(root, noise_type, noise_rate, tuning=False):
    if noise_type == 'sym':
        T = get_sym_T(noise_rate, 100)
    elif noise_type == 'asym':
        T = get_asym_T_cifar100(noise_rate)
    else:
        raise ValueError('Wrong noise type! Must be sym or asym')
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    
    if tuning:
        train_dataset = MyCIFAR100(root=root,
                                   train=True,
                                   transform=train_transform,
                                   trans_matrix=T)
        num_train = int(len(train_dataset) * 0.9)
        num_eval = len(train_dataset) - num_train
        train_dataset, _ = random_split(train_dataset, [num_train, num_eval],
                                        generator=torch.Generator().manual_seed(42))
        train_dataset.trans_matrix = T

        eval_dataset = MyCIFAR100(root=root,
                                  train=True,
                                  transform=eval_transform)
        _, eval_dataset = random_split(eval_dataset, [num_train, num_eval],
                                       generator=torch.Generator().manual_seed(42))

    else:
        train_dataset = MyCIFAR100(root=root,
                                   train=True,
                                   transform=train_transform,
                                   trans_matrix=T)
        
        eval_dataset = MyCIFAR100(root=root,
                                  train=False,
                                  transform=eval_transform)

    return train_dataset, eval_dataset

def webvision(train_data_path, val_data_path):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
    # use "mini" setting

    train_dataset = WebVisionDataset(root=train_data_path,
                                     file_name='webvision_mini_train.txt',
                                     transform=train_transform)

    eval_dataset = ImageNetMini(root=val_data_path,
                                split='val',
                                transform=test_transform)
    
    return train_dataset, eval_dataset
