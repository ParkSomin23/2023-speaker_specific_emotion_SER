#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:05:47 2022

@author: ubuntu
"""
from cmath import isnan
from xml.dom import ValidationErr
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from collections import Counter

class HS_Loss(nn.Module):
    def __init__(self, out_class, hparams, in_channel=None):
        super(HS_Loss, self).__init__()
        # https://github.com/TreB1eN/InsightFace_Pytorch/blob/350ff7aa9c9db8d369d1932e14d2a4d11a3e9553/model.py
        
        if in_channel is None:
            in_feat = hparams['fin_channel']
        else:
            in_feat = in_channel
        
        self.fc = nn.Parameter(torch.Tensor(out_class, in_feat))

        self.s = hparams['scale']
        self.m = hparams['margin']

        self.m_cos = math.cos(self.m)
        self.m_sin = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = self.m_sin * self.m

        self.eps = 1e-6
        #self.fc.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        nn.init.xavier_uniform_(self.fc.data)
        
    def forward(self, x, labels):
        
        assert len(x) == len(labels)
        nB = len(x)

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.fc))
        cosine = cosine.clamp(-1, 1)

        # cos(theta + m)
        tmp = (1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        sine = torch.sqrt(tmp)
        phi  = cosine * self.m_cos - sine * self.m_sin
        #if torch.isnan(sine).any():
        #    raise ValueError(sine)

        cond_v = cosine - self.th
        cond_mask = cond_v <= 0
        keep_val = (cosine - self.mm) 
        phi[cond_mask] = keep_val[cond_mask]
        output = cosine * 1.0 
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, labels] = phi[idx_, labels]

        output = output * self.s

        return output


class Two_CELoss(nn.Module):
    def __init__(self, rerange_emo_prob, n_class):
        super(Two_CELoss, self).__init__()

        self.rerange_emo_prob = rerange_emo_prob
        self.n_class = n_class

    def forward(self, x, ans):

        prob = F.softmax(x, dim=-1)

        neu_prob = torch.stack([prob[:, 0], 1-prob[:, 0]], dim=-1)
        neu_ans = torch.where(ans == 0, 0, 1).reshape(-1)
        neu_ans = F.one_hot(neu_ans, num_classes=2)
        neu_loss = -1 * (torch.log(neu_prob + 1e-8) * neu_ans)
        neu_loss = torch.sum(neu_loss) / len(prob)

        emo_prob = prob[:, 1:]
        if self.rerange_emo_prob:
            #emo_prob = F.softmax(emo_prob, dim=-1)
            emo_prob = emo_prob / emo_prob.sum(dim=-1).reshape(-1,1)

        emo_mask = torch.where(ans==0, False, True).reshape(-1)
        emo_ans = (ans[emo_mask] - 1).reshape(-1).long()
        emo_ans = F.one_hot(emo_ans, num_classes=(self.n_class-1))
        emo_loss = -1 * (torch.log(emo_prob[emo_mask] + 1e-8) * emo_ans)
        emo_loss = torch.sum(emo_loss) / len(prob)

        loss = emo_loss + neu_loss

        return loss, prob



class SupConLoss(nn.Module):
    """
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    """

    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.eps = 1e-8

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        labels_set = list(set(labels.clone().detach().cpu().reshape(-1).tolist()))
        labels = torch.tensor([labels_set.index(l) for l in labels]).reshape(-1,1).to(device)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[0]
        contrast_feature = features.view(batch_size, -1) #torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #mean_log_prob_pos = mean_log_prob_pos[~mean_log_prob_pos.isnan()]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size)
        loss = loss[~loss.isnan()]
        loss = loss.mean()

        return loss