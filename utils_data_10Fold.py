import pandas as pd
import random
from collections import Counter
import math
import numpy as np
import my_utils
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import copy
import utils_sampler

import torch
import pickle

def real_5Fold(hparams):

    seed = hparams['seed']
    my_utils.set_seed(seed)

    # DATA LOADER
    csv = pd.read_csv(hparams['data_file']) # (csv, p, seed, data_mode, target_class, use_class):
    if hparams['concat_ExcHap']:
        csv = csv.replace('exc', 'hapexc')
        csv = csv.replace('hap', 'hapexc')

    if hparams['concat_FruAng']:
        csv = csv.replace('fru', 'angfru')
        csv = csv.replace('ang', 'angfru')

    mask = [True if e in hparams['use_class'] else False for e in csv['emotion']]
    csv = csv[mask]
    csv = csv.reset_index(drop=True)

    testset = csv[csv['fold'] == hparams['fold_num']]
    if hparams['data_mode'] != 'all':
        test_mask = np.where(testset['mode'] == hparams['data_mode'], True, False)
        testset = testset[test_mask].reset_index(drop=True)
    else:
        testset = testset.reset_index(drop=True)

    # TRAIN & VALID
    dataset = csv[csv['fold'] != hparams['fold_num']]
    if hparams['data_mode'] != 'all':
        data_mask = np.where(dataset['mode'] == hparams['data_mode'], True, False)
        dataset = dataset[data_mask].reset_index(drop=True)
    else:
        dataset = dataset.reset_index(drop=True)

    n_sample = math.ceil(Counter(dataset['emotion']).most_common()[-1][1] * 0.2)
    valid_list  = []
    for e in hparams['use_class']:
        # valid
        index = dataset[dataset['emotion'] == e].index.tolist()
        sample = random.sample(index, n_sample)
        valid_list.extend(sample)  

    validset = dataset.iloc[valid_list]
    validset = validset.reset_index(drop=True)

    train_list = list(set(dataset.index.tolist()) - set(valid_list))
    trainset = dataset.iloc[train_list]
    trainset = trainset.reset_index(drop=True)

    return trainset, validset, testset

def Get_10_Loaders(hparams, Dataset_Class, Pad_Collate):
    
    seed = hparams['seed']
    my_utils.set_seed(seed)

    # DATA LOADER
    csv = pd.read_csv(hparams['data_file']) # (csv, p, seed, data_mode, target_class, use_class):
    if hparams['concat_ExcHap']:
        csv = csv.replace('exc', 'hapexc')
        csv = csv.replace('hap', 'hapexc')

    if hparams['concat_FruAng']:
        csv = csv.replace('fru', 'angfru')
        csv = csv.replace('ang', 'angfru')

    mask = [True if e in hparams['use_class'] else False for e in csv['emotion']]
    csv = csv[mask]
    csv = csv.reset_index(drop=True)

    testset = csv[csv['id_num'] == hparams['fold_num']]
    if hparams['data_mode'] != 'all':
        test_mask = np.where(testset['mode'] == hparams['data_mode'], True, False)
        testset = testset[test_mask].reset_index(drop=True)
    else:
        testset = testset.reset_index(drop=True)
    test_set = Dataset_Class(testset, hparams)
    testloader = DataLoader(test_set, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)


    if hparams['fold_num'] % 2 == 0:
        valid_num = hparams['fold_num'] - 1
    else:
        valid_num = hparams['fold_num'] + 1

    validset = csv[csv['id_num'] == valid_num]
    if hparams['data_mode'] != 'all':
        valid_mask = np.where(validset['mode'] == hparams['data_mode'], True, False)
        validset = validset[valid_mask].reset_index(drop=True)
    else:
        validset = validset.reset_index(drop=True)

    # VALIDSET SAMPLE
    idx_list  = {'train':[], 'valid':[]}

    v_sample = Counter(validset['emotion']).most_common[-1][1]
    if hparams['equal_sampling']:
        for e in hparams['class_lilst']:
            # valid
            index = csv[csv['emotion'] == e].index.tolist()
            sample = random.sample(index, v_sample)
            idx_list['valid'].extend(sample)  

    validset = validset.iloc[idx_list['valid']]
    validset = validset.reset_index(drop=True)

    valid_set = Dataset_Class(validset, hparams)
    validloader = DataLoader(valid_set, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)

    session = math.ceil(hparams['fold_num']/2)
    trainset = csv[csv['fold'] != session]
    if hparams['data_mode'] != 'all':
        train_mask = np.where(trainset['mode'] == hparams['data_mode'], True, False)
        trainset = trainset[train_mask].reset_index(drop=True)
    else:
        trainset = trainset.reset_index(drop=True)
    train_sampler = utils_sampler.ClassSampler(trainset, copy.deepcopy(hparams['use_class']), hparams['batch_sizes'])
    trainset = Dataset_Class(trainset, hparams)
    trainloader = DataLoader(trainset, batch_sampler=train_sampler, collate_fn=Pad_Collate, pin_memory=True)

    return trainloader, validloader, testloader

def TenFold_2_FiveFold(hparams, Dataset_Class, Pad_Collate):
    
    seed = hparams['seed']
    my_utils.set_seed(seed)

    # DATA LOADER
    csv = pd.read_csv(hparams['data_file']) # (csv, p, seed, data_mode, target_class, use_class):
    if hparams['concat_ExcHap']:
        csv = csv.replace('exc', 'hapexc')
        csv = csv.replace('hap', 'hapexc')

    if hparams['concat_FruAng']:
        csv = csv.replace('fru', 'angfru')
        csv = csv.replace('ang', 'angfru')

    mask = [True if e in hparams['use_class'] else False for e in csv['emotion']]
    csv = csv[mask]
    csv = csv.reset_index(drop=True)

    if hparams['data_mode'] != 'all':
        data_mask = np.where(csv['mode'] == hparams['data_mode'], True, False)
        csv = csv[data_mask].reset_index(drop=True)
    else:
        csv = csv.reset_index(drop=True)

    # WHICH ID
    first_ID  = hparams['fold_num'] * 2 - 1 # 1 ~ 5
    second_ID = hparams['fold_num'] * 2

    # first_ID
    f_testset = csv[csv['id_num'] == first_ID]
    if hparams['data_mode'] != 'all':
        test_mask = np.where(f_testset['mode'] == hparams['data_mode'], True, False)
        f_testset = f_testset[test_mask].reset_index(drop=True)
    else:
        f_testset = f_testset.reset_index(drop=True)
    f_test_set = Dataset_Class(f_testset, hparams)
    f_testloader = DataLoader(f_test_set, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)

    f_validset = csv[csv['id_num'] == second_ID]
    f_validset = f_validset.reset_index(drop=True)

    idx_list  = {'train':[], 'valid':[]}
    v_sample = Counter(f_validset['emotion']).most_common()[-1][1]
    if hparams['equal_sampling']:
        for e in hparams['use_class']:
            # valid
            index = f_validset[f_validset['emotion'] == e].index.tolist()
            sample = random.sample(index, v_sample)
            idx_list['valid'].extend(sample)  

    f_validset = f_validset.iloc[idx_list['valid']]
    f_validset = f_validset.reset_index(drop=True)

    f_valid_set = Dataset_Class(f_validset, hparams)
    f_validloader = DataLoader(f_valid_set, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)


    # second_ID
    s_testset = csv[csv['id_num'] == second_ID]
    if hparams['data_mode'] != 'all':
        test_mask = np.where(s_testset['mode'] == hparams['data_mode'], True, False)
        s_testset = s_testset[test_mask].reset_index(drop=True)
    else:
        s_testset = s_testset.reset_index(drop=True)
    s_test_set = Dataset_Class(s_testset, hparams)
    s_testloader = DataLoader(s_test_set, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)

    s_validset = csv[csv['id_num'] == first_ID]
    s_validset = s_validset.reset_index(drop=True)

    idx_list  = {'train':[], 'valid':[]}
    v_sample = Counter(s_validset['emotion']).most_common()[-1][1]
    if hparams['equal_sampling']:
        for e in hparams['use_class']:
            # valid
            index = s_validset[s_validset['emotion'] == e].index.tolist()
            sample = random.sample(index, v_sample)
            idx_list['valid'].extend(sample)  

    s_validset = s_validset.iloc[idx_list['valid']]
    s_validset = s_validset.reset_index(drop=True)

    s_valid_set = Dataset_Class(s_validset, hparams)
    s_validloader = DataLoader(s_valid_set, shuffle=True, batch_size=hparams['batch_sizes'], collate_fn=Pad_Collate, pin_memory=True)

    # TRAINSET
    trainset = csv[csv['fold'] != hparams['fold_num']]
    if hparams['data_mode'] != 'all':
        train_mask = np.where(trainset['mode'] == hparams['data_mode'], True, False)
        trainset = trainset[train_mask].reset_index(drop=True)
    else:
        trainset = trainset.reset_index(drop=True)
    train_sampler = utils_sampler.ClassSampler(trainset, copy.deepcopy(hparams['use_class']), hparams['batch_sizes'])
    train_set = Dataset_Class(trainset, hparams)
    trainloader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=Pad_Collate, pin_memory=True, num_workers=4)

    print(Counter(trainset['emotion']), Counter(trainset['id_num']))
    print('first')
    print(Counter(f_validset['emotion']), Counter(f_validset['id_num']))
    print(Counter(f_testset['emotion']), Counter(f_testset['id_num']))
    print('second')
    print(Counter(s_validset['emotion']), Counter(s_validset['id_num']))
    print(Counter(s_testset['emotion']), Counter(s_testset['id_num']))

    return trainloader, (f_validloader, f_testloader), (s_validloader, s_testloader)


class RandomBackgroundNoise:
    """
    noise_transform = RandomBackgroundNoise(sample_rate, './noises_directory')
    transformed_audio = noise_transform(audio_data)
    """
    def __init__(self, sample_rate, noise_dir, min_snr_db=5, max_snr_db=20):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        self.noise_files = {}
        for sr in self.sample_rate:
            with open(noise_dir + '/'+str(sr)+'_saved_list.pkl', 'rb') as f:
                self.noise_files[sr] = pickle.load(f)

    def __call__(self, audio_data, sr):
        noise = random.choice(self.noise_files[sr])
        noise = noise.reshape(1, -1)
        noise = torch.Tensor(noise)
        
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length-audio_length)
            noise = noise[..., offset:offset+audio_length]
        elif noise_length < audio_length:
            tmp = torch.zeros(audio_data.shape)
            offset = random.randint(0, audio_length-noise_length)
            tmp[..., offset:offset+noise_length] = noise
            noise = tmp

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power
        
        audio_data = audio_data[:, :audio_length]
        noise = noise[:, :audio_length]

        return (scale * audio_data + noise ) / 2
        # '/home/ubuntu/Desktop/MY_GITHUB/re_SDR/musan'