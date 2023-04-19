from email.policy import strict
import glob
import json
import math
import os
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from collections import OrderedDict
import torch.nn.functional as F

import random
from collections import Counter

import pandas as pd

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True



def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

def opt_threshold(tpr, fpr, threshold):
    
    optimal_threshold = np.zeros(2)
    index = np.zeros(2)
    
    # min
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]] #ix
    optimal_threshold[0] = list(roc_t['threshold'])[0]
    index[0] = np.array(roc_t.index)
    
    # max
    gmeans = np.sqrt(tpr*(1-fpr))
    index[1] = int(gmeans.argmax())
    # print(int(index[1]))
    optimal_threshold[1] = threshold[int(index[1])]

    return optimal_threshold, index

def accuracy(pred, ans, threshold):

    threshold_pred = []
    for idx, m in enumerate(pred):
        if m >= threshold:
            threshold_pred.append(1)
        else:
            threshold_pred.append(0)

    threshold_pred = np.array(threshold_pred).reshape(-1, 1)
    ans = np.array(ans).reshape(-1, 1)

    correct = (threshold_pred == ans).sum()
    acc = correct / len(ans)

    return acc


def update_learning_rate(optimizer, new_lr, param_group=None):
    # Iterate all groups if none is provided
    if param_group is None:
        groups = range(len(optimizer.param_groups))

    for i in groups:
        old_lr = optimizer.param_groups[i]["lr"]

        # Change learning rate if new value is different from old.
        if new_lr != old_lr:
            optimizer.param_groups[i]["lr"] = new_lr
            optimizer.param_groups[i]["prev_lr"] = old_lr
            #print("Changing lr from %.2g to %.2g" % (old_lr, new_lr))

def warmup_learning_rate(optimizer, max_lr):

    groups = range(len(optimizer.param_groups))
    for i in groups:
        old_lr = optimizer.param_groups[i]["lr"]
        new_lr = old_lr * 10
        if (old_lr >= max_lr):
            new_lr = max_lr

        # Change learning rate if new value is different from old.
        if new_lr != old_lr:
            optimizer.param_groups[i]["lr"] = new_lr
            optimizer.param_groups[i]["prev_lr"] = old_lr
            #print("Changing lr from %.2g to %.2g" % (old_lr, new_lr))

class NewBobScheduler:
    def __init__(self, initial_lr, improvement_threshold, annealing_factor, patient):
        self.current_lr = initial_lr
        self.improvement_threshold = improvement_threshold
        self.annealing_factor = annealing_factor
        self.patient = patient
        self.metric_list = []
        self.current_patient = self.patient

    def __call__(self, metric_value):

        old_lr = new_lr = self.current_lr
        if len(self.metric_list) > 0:
            prev_metric = self.metric_list[-1]

            if prev_metric == 0:
                improvement = 0
            else:
                improvement = (prev_metric - metric_value) / prev_metric

            if improvement < self.improvement_threshold:
                if self.current_patient == 0:
                    new_lr = self.annealing_factor * new_lr
                    self.current_patient = self.patient
                else:
                    self.current_patient -= 1

        self.metric_list.append(metric_value)
        self.current_lr = new_lr

        return old_lr, new_lr

    def save(self, path):
        data = {'current_lr': self.current_lr,
                'current_patient': self.current_patient,
                'metric_list': self.metric_list}
        torch.save(data, path)

    def load(self, path):

        data = torch.load(path)
        self.current_lr = data['current_lr']
        self.current_patient = data['current_patient']
        self.metric_list = data['metric_list']

        del data

class LinearWarmUpScheduler:
    def __init__(self, max_warmup, min_warmnp, num_warmup, annling):
        
        self.current_lr = min_warmnp

        self.min_warmnp = min_warmnp
        self.max_warmup = max_warmup
        self.num_warmup = num_warmup
        
        if self.num_warmup == 0:
            self.current_lr = self.max_warmup
        else:
            self.warmup_value = (self.max_warmup - self.min_warmnp) / self.num_warmup        
        self.annling = annling

        self.current_warmup = 1
        self.metric_list = []

    def __call__(self):

        old_lr = self.current_lr
        if self.current_warmup <= self.num_warmup:
            new_lr = self.current_lr + self.current_warmup
            self.current_warmup += 1

            #self.metric_list.append(metric_value)
            self.current_lr = new_lr

            return old_lr, new_lr

        else:
            new_lr = old_lr * self.annling
            self.current_lr = new_lr

            return old_lr, new_lr

    def save(self, path):
        data = {'current_lr': self.current_lr,
                'current_warmup': self.current_warmup,}
                #'metric_list': self.metric_list}
        torch.save(data, path)

    def load(self, path):

        data = torch.load(path)
        self.current_lr = data['current_lr']
        self.current_patient = data['current_warmup']
        #self.metric_list = data['metric_list']

        del data


