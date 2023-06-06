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
            pid = 0

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

class Emotion_Network(nn.Module):
    def __init__(self, hparams):
        super(Emotion_Network, self).__init__()

        self.hparams = hparams

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

        x, norm_x = self.w2v.extract_features(x)

        x = self.relu(x[-1])
        x = self.fc1(x)

        out_list = []
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

        loss = F.cross_entropy(out, ans.reshape(-1).long())

        return loss, [out], [ans], torch.zeros(1)
    
    def get_feat(self, x):
        batch, _ = x.size()

        with torch.no_grad():

            x, norm_x = self.w2v.extract_features(x)

            x = self.relu(x[-1])
            x = self.fc1(x)

            feat_list = []
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

            return out, feat, feat_list

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
    running_loss = 0

    answ_dict = []
    pred_dict = []
    pid_dict = []

    network.eval()
    for inputs, ans, pid in tqdm(dataloader):
        inputs = inputs.to(device=device)
        ans = torch.Tensor(ans).to(device=device)

        with torch.no_grad():
            
            out, _, _ = network.get_feat(inputs)
            
            loss = F.cross_entropy(out, ans.reshape(-1).long())

            running_loss += loss.item()
            
            p = out.clone().detach().cpu().argmax(dim=-1).reshape(-1)
            a = ans.clone().detach().cpu().reshape(-1)

            answ_dict.extend(a)
            pred_dict.extend(p)
            pid_dict.extend(pid)

            i += 1

    total_loss = running_loss / i

    acc_dict = {'UA':[], 'WA':[]}
    acc_dict['UA'] = get_pred_acc(pred_dict, answ_dict, balanced_accuracy=True)
    acc_dict['WA'] = get_pred_acc(pred_dict, answ_dict, balanced_accuracy=False)

    return total_loss, acc_dict, answ_dict, pred_dict, pid_dict

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

    train_sampler = utils_sampler.ClassSampler(trainset, copy.deepcopy(hparams['use_class']), hparams['batch_sizes'])
    trainset = IEMOCAP_Dataset(trainset, hparams)
    trainloader = DataLoader(trainset, batch_sampler=train_sampler, collate_fn=Pad_Collate, pin_memory=True)

    net = Emotion_Network(hparams)

    for n, p in net.w2v.named_parameters():
        if 'feature_extractor' in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

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
            ##### TRAIN #####
            running_loss = 0
            emo_batch_loss = 0

            answ_dict = []
            pred_dict = []
            
            net.train()
            tbar = tqdm(trainloader)
            for _, (inputs, labels, ids) in enumerate(tbar):

                inputs = inputs.to(device=device)
                labels = torch.Tensor(labels).to(device=device)
                ids = torch.Tensor(ids).to(device=device)
                
                # loss, [out], [ans], id_pred
                emo_loss, output, answers, id_pred = net(inputs, labels, ids)

                emo_batch_loss += emo_loss.item()
                
                loss = emo_loss / hparams['accum_grad']
                loss.backward()
                if (i+1) % hparams['accum_grad'] == 0 or ((i+1)==trainloader.batch_sampler.length):

                    grad = my_utils.get_grad_norm(net.parameters(), 1)
                    clip_grad = nn.utils.clip_grad_norm_(net.parameters(), hparams['max_grad'])
                    wandb.log({'grad': grad})

                    if torch.isnan(clip_grad):
                        print(f'grad is nan')
                    else:
                        optimizer.step()        
                    optimizer.zero_grad()     

                p = output[0].clone().detach().cpu().argmax(dim=-1).reshape(-1)
                a = answers[0].clone().detach().cpu().reshape(-1)

                answ_dict.extend(a)
                pred_dict.extend(p)
                
                i += 1
                
                running_loss += loss.item()

                tbar.set_description('epoch: {}'.format(epoch))
                tbar.set_postfix({'t_loss' : running_loss / math.ceil(i/hparams['accum_grad'])})
                
            t_loss = running_loss / math.ceil(i/hparams['accum_grad'])
            t_acc  = utils_trainer.get_pred_acc(pred_dict, answ_dict, balanced_accuracy=True)
            
            ##### VALID #####
            # total_loss, acc_dict, answ_dict, pred_dict, feat_dict
            v_loss, acc_dict, _, _, _ = eval_network(net, validloader)
            v_acc = acc_dict['WA']

            scheduler.step()

            ##### PRINT RESULT #####
            t_line = f'[train - {epoch}] loss: {t_loss:.3f} emo_acc: {t_acc:.3f}'
            v_line = f'[valid - {epoch}] loss: {v_loss:.3f} emo_acc: {v_acc: .3f}'            
            print(t_line)
            print(v_line)

            ##### LOG RESULT #####            
            logs = {'train_loss':t_loss,

                    'train_emo_loss':emo_batch_loss/ math.ceil(i/hparams['accum_grad']),

                    'train_emo_acc':t_acc,
                    
                    'valid_loss':v_loss,
                    'valid_emo_acc': v_acc
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
            
        best_state = torch.load(save_folder+'/best_model.pt')
        net.load_state_dict(best_state['model_state_dict'])

        #eer, minDCF = SpeakerVerification(hparams, net, hparams['test_path'], hparams['test_file'])
        _, acc_dict, _, _, _ = eval_network(net, testloader)
        acc = acc_dict['WA']

        wandb.run.summary["test_WA"] = acc_dict['WA']
        wandb.run.summary["test_UA"] = acc_dict['UA']

        fin_print = f'[{best_epoch }] acc: {acc:.3f}'
        print(fin_print)

        wandb.finish()

    except KeyboardInterrupt:

        best_state = torch.load(save_folder+'/best_model.pt')
        net.load_state_dict(best_state['model_state_dict'])

        _, acc_dict, _, _, _ = eval_network(net, testloader)
        acc = acc_dict['WA']

        wandb.run.summary["test_WA"] = acc_dict['WA']
        wandb.run.summary["test_UA"] = acc_dict['UA']

        fin_print = f'[{best_epoch }] acc: {acc:.3f}'
        print(fin_print)

        wandb.finish()
   

if __name__ == '__main__':

    # pip install wandb --upgrade
    for n_pool in [3]:
        torch.cuda.empty_cache()

        hparams = {}
        # 
        hparams['pj_name'] = 'Vox_IEMO_REVISION'
        hparams['memo'] = 'CE_only_emo'

        hparams['repeat_3_sec'] = True # True/False

        hparams['emo_hs_linear'] = 'linear'

        hparams['alpha_loss'] = 1
        hparams['beta_loss'] = 1

        hparams['start_patience_epoch'] = 50
        hparams['patience_limit'] = 10

        # w2v config
        hparams['fin_channel'] = 768
        hparams['pool'] = 'attn_max'
        hparams['pool_head'] = n_pool

        # HS Loss
        hparams['emo_scale'] = 30
        hparams['emo_margin'] = 0.3
        
        # train
        hparams['total_epochs'] = 150
        hparams['accum_grad'] = 4

        hparams['cuda_num'] = 0
        hparams['seed'] = 1000

        hparams['batch_sizes'] = 6
        hparams['lr'] = 1e-5*3 #if fold == 2 else 1e-5*3
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
            hparams['data_file'] = '/home/cvnar2/Desktop/ssd/somin/vad_net/emo_vad_sec.csv'

        for fold in range(1,6):
            hparams['fold_num'] = fold

            folder_name = os.path.basename(__file__).split('.')[0]
            hparams_path = folder_name + '_train_hparams.yaml'
            with open(hparams_path, 'w') as f:
                yaml.dump(hparams, f, sort_keys=False)

            main(hparams_path)

        
        
