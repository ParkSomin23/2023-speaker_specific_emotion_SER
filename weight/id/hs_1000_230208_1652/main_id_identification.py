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
import warnings
import loss_class
import itertools

#import clova_ai.tuneThreshold as tuneThreshold

def mk_savefolder(cwd, basename, pj_name, hparams, hparam_path, current_time):
    main_folder = cwd + '/'+ pj_name + '/' + basename.split('.')[0]
    if not os.path.isdir(main_folder):
        os.mkdir(main_folder)

    save_folder = main_folder + '/' + '_'.join([str(hparams['seed']), current_time])

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

class Vox_Dataset(Dataset):
    def __init__(self, vox_df, max_sec):

        self.max_sec = max_sec
        self.vox_df = vox_df

        self.id_list = list(set(vox_df['id']))

        self.t_set = set(os.listdir('/media/ubuntu/SSD2/Dataset/voxceleb/vox1_test_wav/'))

    def __getitem__(self, index):

        data = self.vox_df.iloc[index]

        path = data['path']
        id = 'id' + str(data['id'])

        if id in self.t_set:
            w_path = '/media/ubuntu/SSD2/Dataset/voxceleb/vox1_test_wav/' + path            
        else:
            w_path = '/media/ubuntu/SSD2/Dataset/voxceleb/voxceleb1/' + path

        wav, sr = torchaudio.load(w_path)
        wav = wav.squeeze(0)
        if (len(wav)/16000 > self.max_sec):
            max_len = int(16000 * self.max_sec)
            offset = random.randint(0, wav.shape[0] - max_len - 1)
            wav = wav[offset:offset+max_len]

        id_num = data['id']
        target = self.id_list.index(id_num)
        
        return wav, target

    def __len__(self):
        return len(self.vox_df)

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
        
    return padded_inputs, labels

class Current_Network(nn.Module):
    def __init__(self, hparams, n_class):
        super(Current_Network, self).__init__()

        self.hparams = hparams

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.w2v = bundle.get_model()

        self.pool = pool_module.AttentionalPool(hparams['fin_channel'], hparams['pool_head'], 0.1, hparams['pool'].split('_')[-1])
        if hparams['hs_linear'] == 'linear':
            self.hs = nn.Linear(hparams['fin_channel'], n_class)
        elif hparams['hs_linear'] == 'hs':
            self.hs = loss_class.HS_Loss(n_class, hparams)
        else:
            raise ValueError('hs_linear value error')
        
    def forward(self, x, ans):

        batch, _ = x.size()
        x, norm_x = self.w2v.extract_features(x)

        x = self.pool(x[-1])
        out = x.view(batch, -1)

        if hparams['hs_linear'] == 'linear':    
            logit = self.hs(out)
        elif hparams['hs_linear'] == 'hs':
            logit = self.hs(out, ans.reshape(-1).long())
        else:
            raise ValueError('hs_linear value error')

        loss = F.cross_entropy(logit, ans.reshape(-1).long())

        return loss, [logit], [ans], [out]

    def get_feat(self, x):

        batch, _ = x.size()
        x, norm_x = self.w2v.extract_features(x)

        x = self.pool(x[-1])
        out = x.view(batch, -1)

        return out

    def get_close_id(self, x):

        batch, _ = x.size()
        x, norm_x = self.w2v.extract_features(x)

        x = self.pool(x[-1])
        out = x.view(batch, -1)

        cosine = F.linear(F.normalize(out), F.normalize(self.hs.fc))

        return out, cosine

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
    feat_dict = []

    network.eval()
    for inputs, ids in tqdm(dataloader):
        inputs = inputs.to(device=device)
        ids = torch.Tensor(ids).to(device=device)

        with torch.no_grad():
            feat, cosine_value = network.get_close_id(inputs)

            loss = F.cross_entropy(cosine_value, ids.reshape(-1).long())
            running_loss += loss.item()
            
            p = cosine_value.clone().detach().cpu().argmax(dim=-1).reshape(-1)
            a = ids.clone().detach().cpu().reshape(-1)

            answ_dict.extend(a)
            pred_dict.extend(p)
            feat_dict.append(feat.detach().cpu().numpy())

            i += 1

    total_loss = running_loss / i

    acc_dict = {'UA':[], 'WA':[]}
    acc_dict['UA'] = get_pred_acc(pred_dict, answ_dict, balanced_accuracy=True)
    acc_dict['WA'] = get_pred_acc(pred_dict, answ_dict, balanced_accuracy=False)

    return total_loss, acc_dict, answ_dict, pred_dict, feat_dict


def main(hparam_path):

    device = 'cuda'

    with open(hparam_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams['cuda_num'])

    now = datetime.now()
    current_time = now.strftime("%y%m%d_%H%M")
    basename = os.path.basename(__file__)

    id = '_'.join(['iden', hparams['hs_linear'], current_time])

    seed = hparams['seed']
    my_utils.set_seed(seed)

    df = pd.read_csv('/media/ubuntu/SSD2/Dataset/voxceleb/vox1_identification.csv')

    train_df = df[df['num']==1]
    train_df = train_df.reset_index(drop=True)

    valid_df = df[df['num']==2]
    valid_df = valid_df.reset_index(drop=True)

    test_df  = df[df['num']==3]
    test_df = test_df.reset_index(drop=True)

    trainset = Vox_Dataset(train_df, hparams['max_sec'])
    validset = Vox_Dataset(valid_df, hparams['max_sec'])
    testset  = Vox_Dataset(test_df, hparams['max_sec'])

    trainloader = DataLoader(trainset, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True, num_workers=4)
    validloader = DataLoader(validset, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True, num_workers=4)
    testloader  = DataLoader(testset,  shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate)

    net = Current_Network(hparams, len(trainset.id_list))
    for n, p in net.named_parameters():
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
            answ_dict = []
            pred_dict = []
            
            net.train()
            tbar = tqdm(trainloader)
            for inputs, labels in tbar:

                inputs = inputs.to(device=device)
                labels = torch.Tensor(labels).to(device=device)

                loss, output, answers, feat = net(inputs, labels)
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
            t_line = f'[train - {epoch}] loss: {t_loss:.3f} acc: {t_acc:.3f}'
            v_line = f'[valid - {epoch}] loss: {v_loss:.3f} acc: {v_acc: .3f}'            
            print(t_line)
            print(v_line)

            ##### LOG RESULT #####            
            logs = {'train_loss':t_loss,
                    'train_acc':t_acc,
                    
                    'valid_loss':v_loss,
                    'valid_acc': v_acc
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

        fin_print = f'[{best_epoch }] acc: {acc:.3f}'
        print(fin_print)

        wandb.finish()

    except KeyboardInterrupt:

        best_state = torch.load(save_folder+'/best_model.pt')
        net.load_state_dict(best_state['model_state_dict'])

        _, acc_dict, _, _, _ = eval_network(net, testloader)
        acc = acc_dict['WA']

        fin_print = f'[{best_epoch }] acc: {acc:.3f}'
        print(fin_print)

        wandb.finish()
   

if __name__ == '__main__':

    # pip install wandb --upgrade
    
    torch.cuda.empty_cache()
    #w2v_config_path = '/home/ubuntu/Desktop/hs_정리/w2v_config.yaml'
    #with open(w2v_config_path) as f:
    #    hparams = yaml.load(f, Loader=yaml.FullLoader)

    hparams = {}
    # 
    hparams['memo'] = 'iden'
    hparams['pj_name'] = 'Vox'

    hparams['hs_linear'] = 'hs'
    hparams['patience_limit'] = 5

    # w2v config
    hparams['fin_channel'] = 768

    # network
    hparams['pool'] = 'attn_max'
    hparams['pool_head'] = 1

    # HS Loss
    hparams['scale'] = 30
    hparams['margin'] = 0.3
    
    # train
    hparams['total_epochs'] = 50
    hparams['accum_grad'] = 2

    hparams['cuda_num'] = 0
    hparams['seed'] = 1000

    hparams['batch_sizes'] = 16

    hparams['lr'] = 1e-5*3 #if fold == 2 else 1e-5*3
    hparams['weight_decay'] = 1e-8
    hparams['max_grad'] = 100.0

    hparams['adam_beta1'] = 0.9
    hparams['adam_beta2'] = 0.98
    hparams['adam_eps'] = 1e-6

    hparams['gamma'] = 0.98
   
    # data
    hparams['max_sec'] = 15
    
    folder_name = os.path.basename(__file__).split('.')[0]
    hparams_path = folder_name + '_train_hparams.yaml'
    with open(hparams_path, 'w') as f:
        yaml.dump(hparams, f, sort_keys=False)

    main(hparams_path)

    
    