import copy
import glob
import math
import os
import pickle
import random
from collections import Counter, OrderedDict
from datetime import datetime
from shutil import copyfile
from turtle import forward

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

import utils_trainer
import utils_sampler
import utils_data_10Fold
import warnings
import loss_class
import itertools

def mk_savefolder(cwd, basename, pj_name, hparams, hparam_path, current_time):
    main_folder = cwd + '/'+ pj_name + '/' + basename.split('.')[0]
    if not os.path.isdir(main_folder):
        os.mkdir(main_folder)

    save_folder = main_folder + '/' + '_'.join([str(hparams['fold_num']), current_time])

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    copyfile(cwd + '/' + basename, save_folder+'/'+basename)
    copyfile(hparam_path, save_folder+'/hparams.yaml')
    copyfile(cwd + '/my_utils.py', save_folder+'/my_utils.py')
    copyfile(cwd + '/utils_sampler.py', save_folder+'/utils_sampler.py')
    copyfile(cwd + '/utils_trainer.py', save_folder+'/utils_trainer.py')
    copyfile(cwd + '/loss_class.py', save_folder+'/loss_class.py')
    copyfile(cwd + '/pool_module.py', save_folder+'/pool_module.py')

    return save_folder

class IEMOCAP_Dataset(Dataset):
    def __init__(self, data, hparams):
        
        self.hparams = hparams
        self.csv = data
        self.output_class = hparams['use_class']
        self.vad = hparams['vad']
        self.sr = 16000

        self.aranged_id_num = list(range(1,11))
        self.aranged_id_num.remove(2*hparams['fold_num']-1)
        self.aranged_id_num.remove(2*hparams['fold_num'])
    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        data = self.csv.iloc[idx]
        ids = data['session']
        ans = self.output_class.index(data['emotion'])

        try:
            pid = self.aranged_id_num.index(data['id_num'])
        except:
            pid = -1

        if self.vad:
            wav_path = '/media/ubuntu/SSD2/Dataset/IEMOCAP_VAD/Session'+str(data['fold'])\
                        +'/' + '_'.join(ids.split('_')[:-1]) + '/' + ids +'.wav'
        else:
            wav_path = '/media/ubuntu/SSD2/Dataset/IEMOCAP_full_release/Session'+str(data['fold'])\
                         +'/sentences/wav/' + '_'.join(ids.split('_')[:-1]) + '/' + ids +'.wav'
            if os.getcwd().split('/')[2] == 'cvnar2':
                wav_path = wav_path.replace('/media/ubuntu/SSD2/Dataset', '/home/cvnar2/Desktop/ssd')


        wav, sr = torchaudio.load(wav_path, normalize=True)
        if self.hparams['repeat_3_sec']:
            if wav.shape[-1] / self.sr < 3.0:
                n_repeat = int(3 // (wav.shape[-1] / self.sr))
                wav = wav.repeat((1, n_repeat))

        if (wav.shape[-1] / self.sr) > self.hparams['max_sec']:
            max_len = int(self.sr * self.hparams['max_sec'])
            offset = random.randint(0, wav.shape[1] - max_len - 1)
            wav = wav[:, offset:offset+max_len]

        inputs = wav.transpose(0, 1).squeeze(1)
        
        return inputs, ans, pid

def Pad_Collate(samples):
    """
    DataLoader collate_fn
    """
    #for i in range(len(samples)):
    #    print(samples[i][0])
    #    print()
    inputs = [sample[0].squeeze(0) for sample in samples]  
    padded_inputs = pad_sequence(inputs, batch_first=True)

    labels = [sample[1] for sample in samples]
    labels = torch.Tensor(labels).float()
    labels = labels.unsqueeze(1)

    ids = [sample[2] for sample in samples]
    ids = torch.Tensor(ids).float()
    ids = ids.unsqueeze(1)
        
    return padded_inputs, labels, ids

class ID_Network(nn.Module):
    def __init__(self, hparams, n_class):
        super(ID_Network, self).__init__()

        self.hparams = hparams

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.w2v = bundle.get_model()

        self.pool = pool_module.AttentionalPool(hparams['fin_channel'], 4, 0.1, 'max')
        if self.hparams['id_hs_linear'] == 'linear':
            self.hs = nn.Linear(hparams['fin_channel'], n_class)
        elif self.hparams['id_hs_linear'] == 'hs':
            self.hs = loss_class.HS_Loss(n_class, hparams['id_scale'], hparams['id_margin'], hparams['fin_channel'])
        else:
            raise ValueError('hs_linear value error')
        
    def forward(self, x, ans):

        batch, _ = x.size()
        x, _ = self.w2v(x)

        x = self.pool(x)
        feat = x.view(batch, -1)

        if self.hparams['id_hs_linear'] == 'linear':    
            out = self.hs(feat)
        elif self.hparams['id_hs_linear'] == 'hs':
            out = self.hs(feat, ans.reshape(-1).long())
        else:
            raise ValueError('id_hs_linear value error')

        return out, ans, feat

    def get_feat(self, x):

        batch, _ = x.size()
        x, _ = self.w2v(x)

        x = self.pool(x)
        feat = x.view(batch, -1)

        return feat

    def get_close_id(self, x):

        batch, _ = x.size()
        x, _ = self.w2v(x)

        x = self.pool(x)
        feat = x.view(batch, -1)

        if self.hparams['id_hs_linear'] == 'linear':    
            out = self.hs(feat)

        elif self.hparams['id_hs_linear'] == 'hs':
            out = F.linear(F.normalize(feat), F.normalize(self.hs.fc))
            out = out.clamp(-1, 1)

        return feat, out

class Emotion_Network(nn.Module):
    def __init__(self, hparams):
        super(Emotion_Network, self).__init__()

        self.hparams = hparams

        self.id_net = ID_Network(hparams, 1251)

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.w2v = bundle.get_model()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hparams['fin_channel'], hparams['fin_channel'])

        self.pool_layers = [pool_module.AttentionalPool(hparams['fin_channel'], 4, 0.1, 'max') for _ in range(hparams['pool_head'])]
        self.pool_layers = nn.ModuleList(self.pool_layers)

        if self.hparams['emo_hs_linear'] == 'linear':
            self.hs = nn.Linear(hparams['fin_channel'], 4)
        elif self.hparams['emo_hs_linear'] == 'hs':
            self.hs = loss_class.HS_Loss(4, hparams['emo_scale'], hparams['emo_margin'], hparams['fin_channel'])
        else:
            raise ValueError('emo_hs_linear value error')

        self.id_filter = nn.Parameter(torch.randn(1, int(hparams['pool_head']), 768).float())

    def forward(self, x, ans, ids):

        batch, _ = x.size()
        if self.hparams['id_net_freeze'] == 'freeze':
            with torch.no_grad():
            # id_loss, id_out, id_ans, id_feat
                id_feat, id_out = self.id_net.get_close_id(x)
        else:
            # out, ans, feat
            id_out, _, id_feat = self.id_net(x, ids)

        x, _ = self.w2v(x)

        x = self.relu(x)
        x = self.fc1(x)

        out_list = [id_feat]
        for pool_layer in self.pool_layers:
            tmp = pool_layer(x)
            tmp = tmp.view(batch, -1)
            out_list.append(tmp)

        feat = torch.stack(out_list, dim=1)
        # out = torch.stack((id_feat, n_out, a_out, h_out, s_out), dim=1)

        feat = feat * self.id_filter
        feat = feat.sum(dim=1)

        if self.hparams['emo_hs_linear'] == 'linear':    
            out = self.hs(feat)
        elif self.hparams['emo_hs_linear'] == 'hs':
            out = self.hs(feat, ans.reshape(-1).long())
        else:
            raise ValueError('emo_hs_linear value error')

        return out, id_out
    
    def get_feat(self, x):
        batch, _ = x.size()

        with torch.no_grad():

            id_feat, id_out = self.id_net.get_close_id(x) 
            x, norm_x = self.w2v.extract_features(x)

            x = self.relu(x[-1])
            x = self.fc1(x)

            feat_list = [id_feat]
            for pool_layer in self.pool_layers:
                tmp = pool_layer(x)
                tmp = tmp.view(batch, -1)
                feat_list.append(tmp)

            feat = torch.stack(feat_list, dim=1)
            # out = torch.stack((id_feat, n_out, a_out, h_out, s_out), dim=1)

            feat = feat * self.id_filter
            feat = feat.sum(dim=1)

            if self.hparams['emo_hs_linear'] == 'linear':    
                out = self.hs(feat)
            elif self.hparams['emo_hs_linear'] == 'hs':
                out = F.linear(F.normalize(feat), F.normalize(self.hs.fc))
                out = out.clamp(-1, 1)
            else:
                raise ValueError('emo_hs_linear value error')

            return [out, id_out], feat, feat_list


def get_pred_acc(output, answer, balanced_accuracy):

    pred = np.array(output).reshape(-1,1)
    answ = np.array(answer).reshape(-1,1)

    cm = confusion_matrix(answ, pred)
    if balanced_accuracy:
        acc = balanced_accuracy_score(answ, pred)
    else:
        acc = (np.eye(len(cm)) * cm).sum() / cm.sum()

    return acc

def eval_network(network, dataloader, device='cuda'):

    i = 0
    running_id_loss = 0
    running_emo_loss = 0

    id_answ_dict = []
    id_pred_dict = []

    emo_answ_dict = []
    emo_pred_dict = []

    network.eval()
    for inputs, ans, pid in tqdm(dataloader):
        inputs = inputs.to(device=device)
        ans = torch.Tensor(ans).to(device=device)
        pid = torch.Tensor(pid).to(device=device)

        with torch.no_grad():
            
            [emo_out, id_out], _, _ = network.get_feat(inputs)

            if hparams['id_net_freeze'] != 'freeze':
                id_loss  = F.cross_entropy(id_out, pid.reshape(-1).long(), ignore_index=-1) 
                running_id_loss += id_loss.item()

                p = id_out.clone().detach().cpu().argmax(dim=-1).reshape(-1).numpy()
                a = pid.clone().detach().cpu().reshape(-1).numpy()

                id_answ_dict.extend(a)
                id_pred_dict.extend(p)
        
    
            emo_loss = F.cross_entropy(emo_out, ans.reshape(-1).long(), ignore_index=-1)
            running_emo_loss += emo_loss.item()

            p = emo_out.clone().detach().cpu().argmax(dim=-1).reshape(-1).numpy()
            a = ans.clone().detach().cpu().reshape(-1).numpy()

            emo_answ_dict.extend(a)
            emo_pred_dict.extend(p)

        i += 1

    acc_dict = {'id_UA' :[], 'id_WA':[],
                'emo_UA':[], 'emo_WA':[]}
    
    if hparams['id_net_freeze'] != 'freeze':
        id_answ_dict = np.array(id_answ_dict)
        id_pred_dict = np.array(id_pred_dict)

        acc_dict['id_UA'] = get_pred_acc(id_pred_dict, id_answ_dict, balanced_accuracy=True)
        acc_dict['id_WA'] = get_pred_acc(id_pred_dict, id_answ_dict, balanced_accuracy=False)

    emo_answ_dict = np.array(emo_answ_dict)
    emo_pred_dict = np.array(emo_pred_dict)
   
    acc_dict['emo_UA'] = get_pred_acc(emo_pred_dict, emo_answ_dict, balanced_accuracy=True)
    acc_dict['emo_WA'] = get_pred_acc(emo_pred_dict, emo_answ_dict, balanced_accuracy=False)

    return [running_id_loss / i, running_emo_loss / i], acc_dict, [emo_answ_dict, emo_pred_dict]
    

def init_with_pretrained_weight(id_net, tmp_trainset, device='cuda'):
    output_dict = {}
    tmp_loader = DataLoader(tmp_trainset, collate_fn=Pad_Collate, pin_memory=True)

    id_net = id_net.to(device=device)
    id_net.eval()
    for inputs, ans, pid in tqdm(tmp_loader):
        inputs = inputs.to(device=device)
        pid = list(pid.reshape(-1))
        with torch.no_grad():
            feat = id_net.get_feat(inputs)
            for _id, value in zip(pid, feat):
                _id = int(_id.item())
                if _id in output_dict.keys():
                    output_dict[_id].append(value)
                else:
                    output_dict[_id] = []
                    output_dict[_id].append(value)

    init_weight = []
    order_list = list(output_dict.keys())
    order_list.sort()
    for _id in order_list:
        tmp = torch.stack(output_dict[_id], dim=0).mean(dim=0)
        init_weight.append(tmp)

    init_weight = torch.stack(init_weight, dim=0)
    init_weight = init_weight.to(device=device)

    del tmp_loader

    return init_weight

def main(hparam_path):

    device = 'cuda'

    with open(hparam_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams['cuda_num'])

    now = datetime.now()
    current_time = now.strftime("%y%m%d_%H%M")
    basename = os.path.basename(__file__)

    id = '_'.join([hparams['emo_hs_linear'], str(hparams['fold_num']), current_time])

    seed = hparams['seed']
    my_utils.set_seed(seed)
    
    trainset, validset, testset = utils_data_10Fold.real_5Fold(hparams)
    
    test_set = IEMOCAP_Dataset(testset, hparams)
    testloader = DataLoader(test_set, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)

    valid_set = IEMOCAP_Dataset(validset, hparams)
    validloader = DataLoader(valid_set, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)

    #train_sampler = utils_sampler.ClassSampler(trainset, copy.deepcopy(hparams['use_class']), hparams['batch_sizes'])
    trainset = IEMOCAP_Dataset(trainset, hparams)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)

    net = Emotion_Network(hparams)

    emo_weight = torch.load(hparams['emo_weight'])
    missing_keys = net.load_state_dict(emo_weight['model_state_dict'], strict=False)
    for m in missing_keys[0]:
        if 'id' not in m:
            print(m)
    print(missing_keys[1])

    tmp = net.id_filter.data
    net.id_filter.data = nn.Parameter(torch.cat([torch.randn(1, 1, hparams['fin_channel']), tmp], dim=1))

    for n, p in net.w2v.named_parameters():
        if 'feature_extractor' in n:
            p.requires_grad = False
        else:
            p.requires_grad = True    

    id_weight = torch.load(hparams['id_weight'])
    missing_keys = net.id_net.load_state_dict(id_weight['model_state_dict'], strict=True)
    print(missing_keys)

    if hparams['id_net_freeze'] == 'freeze':
        for n, p in net.id_net.named_parameters():
            p.requires_grad = False
    else:
        if hparams['init_id_with_trainset']:
            init_weight = init_with_pretrained_weight(net.id_net, trainset, device)
            net.id_net.hs.fc = nn.Parameter(torch.Tensor(8, hparams['fin_channel']))
            net.id_net.hs.fc.data = init_weight
        else:
            net.id_net.hs.fc = nn.Parameter(torch.Tensor(8, hparams['fin_channel']))
            nn.init.xavier_uniform_(net.id_net.hs.fc.data)

        if hparams['id_net_freeze'] == 'arcface_finetune':
            for n, p in net.id_net.named_parameters():
                if 'w2v' in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

                if hparams['id_pool_freeze'] and ('pool' in n):
                    p.requires_grad = False            
        
        elif hparams['id_net_freeze'] == 'all_finetune':
            for n, p in net.id_net.w2v.named_parameters():
                if 'feature_extractor' in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            
        else:
            raise ValueError('check id_net_freeze value')
    
    net = net.to(device=device)
    
    optimizer = optim.Adam(net.parameters(), lr=hparams['lr'], betas=(hparams['adam_beta1'], hparams['adam_beta2']), \
                            eps=hparams['adam_eps'], weight_decay=hparams['weight_decay'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,  lr_lambda = lambda epoch: hparams['gamma'] ** epoch)

    patience_limit = hparams['patience_limit']

    wandb.init(project=hparams['pj_name'], name=id, id=id, resume=False, config = hparams)
    
    try:
        best_acc = 0
        best_epoch = -1

        best_loss = 1e+5
        patience_check = 0
        
        for epoch in range(hparams['total_epochs']):  # loop over the dataset multiple times
            
            i = 0

            running_loss = 0
            running_id_loss = 0
            running_emo_loss = 0

            id_answ_dict = []
            id_pred_dict = []

            emo_answ_dict = []
            emo_pred_dict = []
            
            net.train()
            tbar = tqdm(trainloader)
            for _, (inputs, ans, ids) in enumerate(tbar):

                inputs = inputs.to(device=device)
                ans = torch.Tensor(ans).to(device=device)
                pid = torch.Tensor(ids).to(device=device)
                
                emo_out, id_out = net(inputs, ans, pid)
            
                if hparams['id_net_freeze'] != 'freeze':
                    id_loss  = F.cross_entropy(id_out, pid.reshape(-1).long(), ignore_index=-1) 
                    
                    p = id_out.clone().detach().cpu().argmax(dim=-1).reshape(-1).numpy()
                    a = pid.clone().detach().cpu().reshape(-1).numpy()

                    id_answ_dict.extend(a)
                    id_pred_dict.extend(p)
                else:
                    hparams['beta_loss'] = 0
                    id_loss = torch.tensor(0).to(device)
            
        
                emo_loss = F.cross_entropy(emo_out, ans.reshape(-1).long(), ignore_index=-1)

                p = emo_out.clone().detach().cpu().argmax(dim=-1).reshape(-1).numpy()
                a = ans.clone().detach().cpu().reshape(-1).numpy()

                emo_answ_dict.extend(a)
                emo_pred_dict.extend(p)


                loss = hparams['alpha_loss'] * emo_loss +  hparams['beta_loss'] * id_loss
                loss = loss / hparams['accum_grad']
                loss.backward()
                if (i+1) % hparams['accum_grad'] == 0 or ((i+1)==len(trainloader)):

                    grad = my_utils.get_grad_norm(net.parameters(), 1)
                    clip_grad = nn.utils.clip_grad_norm_(net.parameters(), hparams['max_grad'])
                    wandb.log({'grad': grad})

                    if torch.isnan(clip_grad):
                        print(f'grad is nan')
                    else:
                        optimizer.step()        
                    optimizer.zero_grad()     

                i += 1
                
                running_loss += loss.item()
                running_emo_loss += emo_loss.item()
                running_id_loss += id_loss.item()

                tbar.set_description('epoch: {}'.format(epoch))
                tbar.set_postfix({'t_loss' : running_loss / math.ceil(i/hparams['accum_grad'])})
                
            t_loss = running_loss / math.ceil(i/hparams['accum_grad'])
            t_acc  = utils_trainer.get_pred_acc(emo_pred_dict, emo_answ_dict, balanced_accuracy=True)
            if hparams['id_net_freeze'] != 'freeze':
                id_acc = utils_trainer.get_pred_acc(id_pred_dict, id_answ_dict, balanced_accuracy=True)
            else:
                id_acc = 0        
            
            ##### VALID #####
            # [running_id_loss / i, running_emo_loss / i], acc_dict, [emo_answ_dict, emo_pred_dict]
            [v_id_loss, v_emo_loss], acc_dict, _ = eval_network(net, validloader)
            v_loss = hparams['alpha_loss'] * v_emo_loss +  hparams['beta_loss'] * v_id_loss
            v_acc = acc_dict['emo_UA']
            if hparams['id_net_freeze'] != 'freeze':
                v_id_acc = acc_dict['id_UA']
            else:
                v_id_acc = 0

            scheduler.step()

            ##### PRINT RESULT #####
            t_line = f'[train - {epoch}] loss: {t_loss:.3f} emo_acc: {t_acc:.3f} id_acc: {id_acc:.3f}'
            v_line = f'[valid - {epoch}] loss: {v_loss:.3f} emo_acc: {v_acc:.3f} id_acc: {v_id_acc:.3f}'            
            print(t_line)
            print(v_line)

            ##### LOG RESULT #####            
            logs = {'train_loss':t_loss,

                    'train_emo_loss':running_emo_loss/ math.ceil(i/hparams['accum_grad']),
                    'train_id_loss':running_id_loss/ math.ceil(i/hparams['accum_grad']),

                    'train_emo_acc':t_acc,
                    'train_id_acc':id_acc,
                    
                    'valid_emo_loss':v_emo_loss,
                    'valid_emo_acc': v_acc,

                    'valid_id_loss':v_loss,
                    'valid_id_acc': v_id_acc
                    }

            if epoch == 0:
                save_folder = mk_savefolder(os.getcwd(), basename, hparams['pj_name'], hparams, hparam_path, current_time)
            
            if best_acc <= v_acc:
                best_acc = v_acc
                state = net.state_dict()
                best_epoch = epoch
                
                torch.save({
                            'epoch': best_epoch,
                            'best_acc': best_acc,
                            'model_state_dict': state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            }, save_folder+'/best_model.pt')

                
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_epoch"] = best_epoch
            else:
                state = net.state_dict()
                torch.save({
                            'epoch': epoch,
                            'best_acc': best_acc,
                            'model_state_dict': state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            }, save_folder+'/latest_model.pt')
            wandb.log(logs)

            if v_loss > best_loss: # loss가 개선되지 않은 경우
                if epoch >= hparams['start_patience_epoch']:
                    patience_check += 1

                if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                    print('early stop')
                    break

            else: # loss가 개선된 경우
                best_loss = v_loss
                patience_check = 0

                state = net.state_dict()
              
                torch.save({
                            'epoch': epoch,
                            'best_acc': best_acc,
                            'model_state_dict': state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            }, save_folder+'/best_loss_model.pt')
            
        best_state = torch.load(save_folder+'/best_model.pt')
        net.load_state_dict(best_state['model_state_dict'])

        #eer, minDCF = SpeakerVerification(hparams, net, hparams['test_path'], hparams['test_file'])
        _, acc_dict, _ = eval_network(net, testloader)
        acc = acc_dict['emo_UA']

        wandb.run.summary["test_WA"] = acc_dict['emo_WA']
        wandb.run.summary["test_UA"] = acc_dict['emo_UA']

        fin_print = f'[{best_epoch }] acc: {acc:.3f}'
        print(fin_print)

        wandb.finish()

    except KeyboardInterrupt:

        best_state = torch.load(save_folder+'/best_model.pt')
        net.load_state_dict(best_state['model_state_dict'])

        _, acc_dict, _ = eval_network(net, testloader)
        acc = acc_dict['emo_UA']

        wandb.run.summary["test_WA"] = acc_dict['emo_WA']
        wandb.run.summary["test_UA"] = acc_dict['emo_UA']

        fin_print = f'[{best_epoch }] acc: {acc:.3f}'
        print(fin_print)

        wandb.finish()
   
   

if __name__ == '__main__':

    # pip install wandb --upgrade
    for state in ['freeze']:
        torch.cuda.empty_cache()

        hparams = {}
        # 
        hparams['pj_name'] = 'Vox_IEMO_REVISION'
        hparams['memo'] = 'emo(sampler)_id_load_noSampler'

        hparams['repeat_3_sec'] = True # True/False

        hparams['emo_hs_linear'] = 'hs'

        hparams['alpha_loss'] = 1
        hparams['beta_loss'] = 1

        hparams['start_patience_epoch'] = 30
        hparams['patience_limit'] = 10

        hparams['init_id_with_trainset'] = True #True/False

        hparams['id_weight'] = '/home/ubuntu/Dropbox/23_for_revision/vox_id/hs_1000_230208_1652/best_model.pt'
        hparams['id_net_freeze'] = 'freeze' #'freeze' 'arcface_finetune', 'all_finetune'
        hparams['id_pool_freeze'] = False #True/False: arcface_finetune 할 때 영향 freeze 안 할 때 기본이 false 

        hparams['id_hs_linear'] = 'hs'
        
        # w2v config
        hparams['fin_channel'] = 768
        hparams['pool'] = 'attn_max'
        hparams['pool_head'] = 4

        # HS Loss
        hparams['id_scale'] = 30
        hparams['id_margin'] = 0.3

        hparams['emo_scale'] = 30
        hparams['emo_margin'] = 0.3
        
        # train
        hparams['total_epochs'] = 100
        hparams['accum_grad'] = 4

        hparams['cuda_num'] = 0
        hparams['seed'] = 1000

        hparams['batch_sizes'] = 6
        hparams['lr'] = 1e-5 #if fold == 2 else 1e-5*3
        hparams['weight_decay'] = 1e-8
        hparams['max_grad'] = 100.0

        hparams['adam_beta1'] = 0.9
        hparams['adam_beta2'] = 0.98
        hparams['adam_eps'] = 1e-6

        hparams['gamma'] = 0.98
    
        # data
        hparams['max_sec'] = 15
        hparams['vad'] = False
        hparams['data_mode'] = 'all'
        hparams['concat_ExcHap'] = True
        hparams['concat_FruAng'] = False
        hparams['use_class'] = ['ang', 'hapexc', 'sad', 'neu']
        hparams['data_file'] = '/home/ubuntu/Desktop/MY_GITHUB/VAD_net/emo_vad_sec.csv'

        if os.getcwd().split('/')[2] == 'cvnar2':
            hparams['vad'] = False
            hparams['id_weight'] = hparams['id_weight'].replace('ubuntu', 'cvnar2')
            hparams['data_file'] = '/home/cvnar2/Desktop/ssd/somin/vad_net/emo_vad_sec.csv'
        
        if (hparams['id_net_freeze'] == 'freeze') or (hparams['id_hs_linear'] != 'hs'):
                hparams['init_id_with_trainset'] = False
        if hparams['id_net_freeze'] == 'freeze':
                hparams['id_pool_freeze'] = True

        for fold in range(1,6):
            hparams['fold_num'] = fold
            # /home/ubuntu/Dropbox/Using Speaker-specific Emotion Representations in wav2vec 2.0-based modules for SER/weight/only_emo/pool_4
            hparams['emo_weight'] = glob.glob('/home/ubuntu/Dropbox/Using Speaker-specific Emotion Representations in wav2vec 2.0-based modules for SER/weight/Vox_IEMO_REVISION/revision_only_emo/' +\
                                              str(fold)+'_*/best_model.pt')[0]

            folder_name = os.path.basename(__file__).split('.')[0]
            hparams_path = folder_name + '_train_hparams.yaml'
            with open(hparams_path, 'w') as f:
                yaml.dump(hparams, f, sort_keys=False)

            main(hparams_path)

        
        