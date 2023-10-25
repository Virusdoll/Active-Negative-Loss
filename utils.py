import json
import os
import random
import string
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt if self.cnt != 0 else 0

def batch_accuracy(outputs, labels):
    _, pred = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (pred == labels).sum().item()
    return correct / total

def log_display(**kwargs):
    display = '|'
    for key, value in kwargs.items():
        display += ' {}:{:5.3f} |'.format(key, value)
    return display

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def get_exp_id(length):
    exp_id = ''
    for _ in range(length):
        choosen_str = random.SystemRandom().choice(string.ascii_uppercase + 
                                                   string.digits)
        exp_id += choosen_str
    return exp_id

def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def get_config(path):
    with open(path) as json_file:
        cfg = json.load(json_file)
    
    return cfg['model'], cfg['loss'], cfg['dataset'], cfg['optim']

def get_model(name, num_classes):
    # Toymodel
    if name == 'toymodel4l':
        from models.toymodel import toymodel4l
        model = toymodel4l()
    elif name == 'toymodel8l':
        from models.toymodel import toymodel8l
        model = toymodel8l()
    # Resnet
    elif name == 'resnet34':
        from models.resnet import resnet34
        model = resnet34(num_classes)
    elif name == 'resnet50':
        # for webvision experiments
        # we need a 50-class classifier
        # with 224x224 input
        from torchvision.models import resnet50
        model = resnet50()
        num_fc_in = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_in, 50)
    # Error
    else:
        raise ValueError('model name error')
    
    return model

def get_loss(name, num_classes, config, train_loader):
    # Loss
    if name == 'ce':
        from loss import ce
        loss = ce()
    elif name == 'fl':
        from loss import fl
        loss = fl(config)
    elif name == 'mae':
        from loss import mae
        loss = mae(config)
    elif name == 'sce':
        from loss import sce
        loss = sce(num_classes, config)
    elif name == 'gce':
        from loss import gce
        loss = gce(num_classes, config)
    
    # Active Passive Loss
    elif name == 'nce_mae':
        from loss import nce_mae
        loss = nce_mae(num_classes, config)
    elif name == 'nce_rce':
        from loss import nce_rce
        loss = nce_rce(num_classes, config)
    elif name == 'nfl_rce':
        from loss import nfl_rce
        loss = nfl_rce(num_classes, config)
    
    # Asymmetric Loss
    elif name == 'nce_agce':
        from loss import nce_agce
        loss = nce_agce(num_classes, config)
    elif name == 'nce_aul':
        from loss import nce_aul
        loss = nce_aul(num_classes, config)
    elif name == 'nce_ael':
        from loss import nce_ael
        loss = nce_ael(num_classes, config)
    
    # Active Negative Loss
    elif name == 'anl_ce':
        from loss import anl_ce
        loss = anl_ce(num_classes, config)
    elif name == 'anl_fl':
        from loss import anl_fl
        loss = anl_fl(num_classes, config)

    # Error
    else:
        raise ValueError('loss name error')
    
    return loss

def get_dataloader(root, config, noise_type, noise_rate, tuning):
    name = config['name']
    if name == 'mnist':
        from dataset import mnist
        train_dataset, eval_dataset = mnist(root, noise_type, noise_rate, tuning)
    elif name == 'cifar10':
        from dataset import cifar10
        train_dataset, eval_dataset = cifar10(root, noise_type, noise_rate, tuning)
    elif name == 'cifar100':
        from dataset import cifar100
        train_dataset, eval_dataset = cifar100(root, noise_type, noise_rate, tuning)
    elif name =='webvision':
        from dataset import webvision
        train_dataset, eval_dataset = webvision(config['train_data_path'],
                                                config['val_data_path'])
    else:
        raise ValueError('dataset name error')
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config['train_batchsize'],
                                  shuffle=True,
                                  num_workers=config['num_workers'])
    
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=config['test_batchsize'],
                                 shuffle=False,
                                 num_workers=config['num_workers'])
    
    return train_dataloader, eval_dataloader

def get_optimizer(name, params, config):
    nesterov = config['nesterov'] \
               if 'nesterov' in config \
               else False
    if name == 'sgd':
        optimizer = optim.SGD(params,
                              lr=config['learning_rate'],
                              momentum=config['momentum'],
                              weight_decay=config['weight_decay'],
                              nesterov=nesterov)
    else:
        raise ValueError('optimizer name error')
    
    return optimizer

def get_scheduler(name, optimizer, config):
    if name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         T_max=config['T_max'],
                                                         eta_min=config['eta_min'])
    elif name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=config['step_size'],
                                              gamma=config['gamma'])
    else:
        raise ValueError('scheduler name error')

    return scheduler