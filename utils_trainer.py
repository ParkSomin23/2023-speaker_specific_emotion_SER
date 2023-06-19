import copy
import glob
import math
import os
import pickle
import random
from collections import Counter, OrderedDict
from datetime import datetime
from shutil import copyfile

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import yaml
from scipy.spatial import distance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

import my_utils
import pool_module
import wandb

import warnings

def mk_savefolder(cwd, basename, pj_name, hparams, hparam_path, current_time):
    main_folder = cwd + '/'+ pj_name + '/' + basename.split('.')[0]
    if not os.path.isdir(main_folder):
        os.mkdir(main_folder)

    save_folder = main_folder + '/' + '_'.join([str(hparams['fold_num']), str(hparams['seed']), current_time])

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    copyfile(cwd + '/' + basename, save_folder+'/'+basename)
    copyfile(hparam_path, save_folder+'/hparams.yaml')
    copyfile(cwd + '/utils.py', save_folder+'/utils.py')
    copyfile(cwd + '/utils_sampler.py', save_folder+'/utils_sampler.py')
    copyfile(cwd + '/utils_trainer.py', save_folder+'/utils_trainer.py')
    copyfile(cwd + '/loss_class.py', save_folder+'/loss_class.py')

    return save_folder

def get_pred_acc(output, answer, balanced_accuracy=False):

    pred = np.array(output).reshape(-1,1)
    answ = np.array(answer).reshape(-1,1)

    cm = confusion_matrix(answ, pred)
    if balanced_accuracy:
        acc = balanced_accuracy_score(answ, pred)
    else:
        acc = (np.eye(len(cm)) * cm).sum() / cm.sum()

    return acc

def GradNorm_Trainer(hparams, current_epoch, task_loss, task_losses, network, optimizer, initial_task_loss_save):

    weighted_task_loss = torch.mul(network.task_weights, task_loss)
    if current_epoch == 0:
        if torch.cuda.is_available():
            initial_task_loss = task_loss.data.cpu()
        else:
            initial_task_loss = task_loss.data
        initial_task_loss = initial_task_loss.numpy()
    else:
        initial_task_loss = initial_task_loss_save

    loss = torch.sum(weighted_task_loss)
            
    loss.backward(retain_graph=True)                                 
    network.task_weights.grad.data = network.task_weights.grad.data * 0.0

    W = network.get_last_shared_layer()
    norms = []
    for i in range(len(hparams['task'])):
        # get the gradient of this task loss with respect to the shared parameters
        gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
        norms.append(torch.norm(torch.mul(network.task_weights[i], gygw[0])))
    norms = torch.stack(norms)
    
    if torch.cuda.is_available():
        loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
    else:
        loss_ratio = task_loss.data.numpy() / initial_task_loss

    inverse_train_rate = loss_ratio / np.mean(loss_ratio)
    
    # compute the mean norm \tilde{G}_w(t) 
    if torch.cuda.is_available():
        mean_norm = np.mean(norms.data.cpu().numpy())
    else:
        mean_norm = np.mean(norms.data.numpy())
        
    # compute the GradNorm loss 
    # this term has to remain constant
    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** hparams['alpha']), requires_grad=False)
    if torch.cuda.is_available():
        constant_term = constant_term.cuda()

    # this is the GradNorm loss itself
    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

    # compute the gradient for the weights
    network.task_weights.grad = torch.autograd.grad(grad_norm_loss, network.task_weights)[0]  
    if torch.isnan(loss):
        print(f'loss is nan')
    else:
        optimizer.step()           
    optimizer.zero_grad()      

    normalize_coeff = len(hparams['task']) / torch.sum(network.task_weights.data, dim=0)
    network.task_weights.data = network.task_weights.data * normalize_coeff       

    # record
    if torch.cuda.is_available():
        task_losses.append(task_loss.data.cpu().numpy())
    else:
        task_losses.append(task_loss.data.numpy())

    task_record = {
        'loss_ratios': np.sum(task_losses[-1] / task_losses[0]),
        'grad_norm_losses': grad_norm_loss.data.cpu().numpy(),
    }

    for i_task, task in enumerate(hparams['task']):
        task_record['task_losses_'+task]  = task_loss[i_task].data.cpu().numpy()
        task_record['task_weights_'+task] = network.task_weights[i_task].data.cpu().numpy()

    #print('type', type(task_record))

    return task_record, loss, initial_task_loss

def eval_network(hparams, network, dataloader, device='cuda', return_mid=False, balanced_accuracy=True):
    
    i = 0
    running_loss = 0

    answ_dict = {k: [] for k in hparams['task']}
    pred_dict = {k: [] for k in hparams['task']}

    network.eval()
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device=device)
        labels = torch.Tensor(labels).to(device=device)

        with torch.no_grad():
            loss, output, answers = network(inputs, labels)
           
            running_loss += loss.item()
            
            for i_task, task in enumerate(hparams['task']):
                p = output[i_task].clone().detach().cpu().argmax(dim=-1).reshape(-1)
                a = answers[i_task].clone().detach().cpu().reshape(-1)

                answ_dict[task].extend(a)
                pred_dict[task].extend(p)

            i += 1

    total_loss = running_loss / i

    acc_dict = {}
    for i_task, task in enumerate(hparams['task']):
        acc_dict[task] = get_pred_acc(pred_dict[task], answ_dict[task], balanced_accuracy=balanced_accuracy)

    if return_mid:
        return total_loss, acc_dict, answ_dict, pred_dict
    else:
        return total_loss, acc_dict

def return_output(hparams, network, dataloader, device='cuda', return_mid=False):
    
    i = 0
    running_loss = 0

    answ_dict = {k: [] for k in hparams['task']}
    pred_dict = {k: [] for k in hparams['task']}
    outp_dict = {k: [] for k in hparams['task']}

    network.eval()
    for inputs, labels, ids in tqdm(dataloader):
        inputs = inputs.to(device=device)
        labels = torch.Tensor(labels).to(device=device)

        with torch.no_grad():
            loss, output, answers = network(inputs, labels)
            if hparams['task_GradNorm']:
                weighted_task_loss = torch.mul(network.task_weights, loss)
                loss = torch.sum(weighted_task_loss)

            running_loss += loss.item()
            
            for i_task, task in enumerate(hparams['task']):
                p = output[i_task].clone().detach().cpu().argmax(dim=-1).reshape(-1)
                a = answers[i_task].clone().detach().cpu().reshape(-1)
                o = output[i_task].clone().detach().cpu()

                answ_dict[task].extend(a)
                pred_dict[task].extend(p)
                outp_dict[task].extend(o)
                

            i += 1

    total_loss = running_loss / i

    acc_dict = {}
    for i_task, task in enumerate(hparams['task']):
        acc_dict[task] = get_pred_acc(pred_dict[task], answ_dict[task], balanced_accuracy=True)

    if return_mid:
        return total_loss, acc_dict, answ_dict, pred_dict, outp_dict
    else:
        return total_loss, acc_dict, outp_dict

def check_emotion_labels(use_class, class_list, concat_ExcHap, concat_FruAng):
    ##if  class_list[0] != 'neu':
    ##    raise ValueError('neutral must be first in the class')

    if concat_ExcHap:
        if 'exc' in use_class:
            use_class.remove('exc')
        if 'exc' in class_list:
            class_list.remove('exc')

        if 'hap' in use_class:
            use_class.remove('hap')
        if 'hap' in class_list:
            class_list.remove('hap')

        if 'hapexc' not in use_class:
            use_class.append('hapexc')
        if 'hapexc' not in class_list:
            class_list.append('hapexc')

    if concat_FruAng:
        if 'fru' in use_class:
            use_class.remove('fru')
        if 'fru' in class_list:
            class_list.remove('fru')

        if 'ang' in use_class:
            use_class.remove('ang')
        if 'ang' in class_list:
            class_list.remove('ang')

        if 'angfru' not in use_class:
            use_class.append('angfru')
        if 'angfru' not in class_list:
            class_list.append('angfru')