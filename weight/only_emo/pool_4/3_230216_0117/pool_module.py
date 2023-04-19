import torch
import torch.nn as nn
import math

class Len_AvgPool(nn.Module):
    def __init__(self):
        super(Len_AvgPool, self).__init__()

        self.eps = 1e-5
        print('Len_AvgPool')
            
    def forward(self, outputs, lens):
        #print(outputs.shape)
        if lens is None:
            return outputs.mean(dim=1) # (B, Seq, Feat) >> (B, Feat)
        else:
            mean = []
            length = lens * outputs.size(1)
            for idx in range(len(outputs)):
                if length[idx] != 1:
                    m  = torch.mean(outputs[idx, :int(length[idx])+1, ...], dim=0)
                else:
                    m  = torch.mean(outputs[idx, :, ...], dim=0)
                mean.append(m)
            mean = torch.stack(mean)
        
        gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
        mean += gnoise

        mean = mean.unsqueeze(1)

        return mean
    
    def _get_gauss_noise(self, mean_shape, device):
        gnoise = torch.randn(mean_shape, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise) #  0 ~ 1
        gnoise = self.eps * ((1-9) * gnoise + 9)  # (0 ~ -8) + 9 => (9, 1)

        return gnoise
        
class Len_Max(nn.Module):
    def __init__(self):
        super(Len_Max, self).__init__()

        print('Len_Max')
            
    def forward(self, outputs, lens):
        #print(outputs.shape)
        if lens is None:
            return outputs.max(dim=1) # (B, Seq, Feat) >> (B, Feat)
        else:
            max_data = []
            length = lens * outputs.size(1)
            for idx in range(len(outputs)):
                if length[idx] != 1:
                    m  = torch.max(outputs[idx, :int(length[idx])+1, ...], dim=0)
                else:
                    m  = torch.max(outputs[idx, :, ...], dim=0)
                max_data.append(m[0])
            max_data = torch.stack(max_data)
        
        max_data = max_data.unsqueeze(1)

        return max_data

class Len_Trim_Avg(nn.Module):
    def __init__(self, q):
        super(Len_Trim_Avg, self).__init__()

        self.q = q

        print('Len_Trim_Avg')
        #raise NotImplementedError("TRIM AVG is not implemented")
            
    def forward(self, outputs, lens):
        #print(outputs.shape)
        if lens is None:
            n = math.ceil(outputs.shape[1] * self.q)

            a_max = torch.quantile(outputs, 1.0-self.q, dim=1, keepdim=True, interpolation='nearest')
            a_min = torch.quantile(outputs, self.q, dim=1, keepdim=True, interpolation='nearest')

            n_max = torch.where(outputs>a_max, True, False).sum(0)
            n_min = torch.where(outputs<a_min, True, False).sum(0)

            x_sum = torch.sum(dim=1) - (a_max * n_max) - (a_min * n_min)
            x_mean = x_sum / (outputs.shape[1] - (n_max + n_min))
        else:
            mean_data = []
            length = lens * outputs.size(1)
            for idx in range(len(outputs)):
                if length[idx] != 1:
                    m  = outputs[idx, :int(length[idx])+1, ...]
                    m = self.trim(m)
                else:
                    m = outputs[idx, :, ...]
                    m = self.trim(m)
                mean_data.append(m)
            mean_data = torch.stack(mean_data) #(B, F)
        
        x_mean = mean_data.unsqueeze(1) #(B, S, F)

        return x_mean

    def trim(self, x):
        n = math.ceil(x.shape[1] * self.q) #(len, feat)

        a_max = torch.quantile(x, 1.0-self.q, dim=0, keepdim=True, interpolation='nearest')
        a_min = torch.quantile(x, self.q, dim=0, keepdim=True, interpolation='nearest')
    
        n_max = torch.where(x>a_max, True, False).sum(0)
        n_min = torch.where(x<a_min, True, False).sum(0)
        
        x = torch.clamp(x, min=a_min, max=a_max)

        x_sum = torch.sum(x, dim=0) - (a_max * n_max) - (a_min * n_min)
        x_mean = x_sum / (x.shape[0] - (n_max+n_min))

        return x_mean

class AttentionalLenPool(nn.Module):
    def __init__(self, feat_channel, nhead, dropout):
        super(AttentionalLenPool, self).__init__()

        print('AttentionalLenPool')

        self.attention = nn.MultiheadAttention(feat_channel, nhead, dropout=dropout, 
                                          bias=True, add_bias_kv=False, 
                                          add_zero_attn=False, kdim=None, vdim=None, 
                                          batch_first=True)
        #model2.AttentionBlock(feat_channel, nhead, dropout)
        self.len_avgpool = Len_AvgPool()

    def forward(self, output, len):

        output = self.attention(output)
        output = self.len_avgpool(output, len)

        return output

class AttentionalPool(nn.Module):
    def __init__(self, feat_channel, nhead, dropout, pool_func):
        super(AttentionalPool, self).__init__()

        #print('AttentionalPool')

        #self.attention = model2.AttentionBlock(feat_channel, nhead, dropout)

        self.attention = nn.MultiheadAttention(feat_channel, nhead, dropout=dropout, 
                                          bias=True, add_bias_kv=False, 
                                          add_zero_attn=False, kdim=None, vdim=None, 
                                          batch_first=True)
        if pool_func == 'mean':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_func == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError('pool func error')

    def forward(self, output):

        output, attn = self.attention(output, output, output)
        output = torch.transpose(output, 1, 2)
        output = self.pool(output)

        return output