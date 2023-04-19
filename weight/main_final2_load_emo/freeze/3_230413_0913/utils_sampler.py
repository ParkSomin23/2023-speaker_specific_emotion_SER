from random import shuffle
import numpy as np
import torch
import random
from collections import Counter
from torch.utils.data.sampler import Sampler

class Class_Len_Sampler(Sampler):
    def __init__(self, data, class_list, bucket_boundaries, batch_size, seed=1000):
        """
        class마다 동일하게
        """
        self.data = data
        self.class_list = class_list

        self.seed = seed

        self.batch_size = batch_size

        counter = Counter(self.data['emotion'])
        
        self.class_count = [counter[k] for k in self.class_list]
        self.min_n = min(self.class_count)

        self.bucket_boundaries = bucket_boundaries

    def __iter__(self):
        
        random.seed(self.seed)
        self.seed += 1

        # 이번 epoch에서 사용할 data idx
        idx_list = []
        for e in self.class_list:
            index = self.data[self.data['emotion']==e].index.tolist()
            sample = random.sample(index, self.min_n)
            idx_list.extend(sample)  

        tmp_df = self.data.iloc[np.array(idx_list)]

        # length bucket
        data_buckets = dict()
        for d_idx in range(len(tmp_df)):
            sec = tmp_df.iloc[d_idx]['sec']
            if sec > self.bucket_boundaries[-1]:
                sec = self.bucket_boundaries[-1]
                
            pid = self.element_to_bucket_id(sec)
            if pid in data_buckets.keys():
                data_buckets[pid].append(d_idx)
            else:
                data_buckets[pid] = [d_idx]

        # to array
        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])
            if len(data_buckets[k]) == 0:
                del data_buckets[k]

        #print(data_buckets.keys())
        
        batch_idx = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            n_split = int(data_buckets[k].shape[0]/self.batch_size)
            if n_split == 0:
                n_split = 1
            batch_idx += (np.array_split(data_buckets[k], n_split))
        shuffle(batch_idx) # shuffle all the batches so they arent ordered by bucket

        # for next(iter) > it only repeat the part under this comment until iter_list ends
        # if it ends it goes back to top
        self.length = len(batch_idx)
        for i in batch_idx: 
            yield i.tolist() 

    def __len__(self):

        return self.min_n * len(self.class_list)
    
    def element_to_bucket_id(self, seq_length):

        boundaries = list(self.bucket_boundaries)

        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]

        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        
        bucket_id = np.min(np.where(conditions_c))
        
        return bucket_id

class LengthSampler(Sampler):
    def __init__(self, wav_len,  
                bucket_boundaries, batch_size):
        self.ind_n_len = wav_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in enumerate(self.ind_n_len):
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])

        for k in data_buckets.keys():
            if len(data_buckets[k]) == 0:
                del data_buckets[k]

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            n_split = int(data_buckets[k].shape[0]/self.batch_size)
            if n_split == 0:
                n_split = 1
            iter_list += (np.array_split(data_buckets[k], 
                          n_split))
        shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.ind_n_len)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

class ClassSampler(Sampler):
    def __init__(self, data, class_list, batch_size, seed=1000):
        """
        class마다 동일하게
        """
        self.data = data
        self.class_list = class_list

        self.seed = seed

        self.batch_size = batch_size

        counter = Counter(self.data['emotion'])
        
        self.class_count = [counter[k] for k in self.class_list]
        self.min_n = min(self.class_count)

    def __iter__(self):
        # mask dataset first
        idx_list = []
        
        random.seed(self.seed)
        self.seed += 1

        for e in self.class_list:
            index = self.data[self.data['emotion']==e].index.tolist()
            sample = random.sample(index, self.min_n)
            idx_list.extend(sample)  
        shuffle(idx_list)
        
        #to_count = self.data.iloc[idx_list]
        #print(Counter(to_count['emotion']))
        
        n_split = int(len(idx_list)/self.batch_size)
        batch_idx = np.array_split(idx_list, n_split)
        self.length = len(batch_idx)
        # for next(iter) > it only repeat the part under this comment, 
        # until iter_list ends. if it ends it goes back to top
        for i in batch_idx: 
            yield i.tolist() # as it was stored in an array

class ClassSampler_Strict(Sampler):
    def __init__(self, data, class_list, batch_size):
        """
        class마다 동일하게
        """
        self.data = data
        self.class_list = class_list

        self.batch_size = batch_size

        counter = Counter(self.data['emotion'])
        
        self.class_count = [counter[k] for k in self.class_list]
        self.min_n = min(self.class_count)

    def __iter__(self):
        # mask dataset first
        idx_dict = []
        split_num = []
        for e in self.class_list:
            index = self.data[self.data['emotion']==e].index.tolist()
            sample = random.sample(index, self.min_n)
            shuffle(sample)

            emo_n_split = int(len(sample)/(self.batch_size/len(self.class_list)))
            emo_batch_idx = np.array_split(sample, emo_n_split)
            idx_dict.append([torch.tensor(e_idx) for e_idx in emo_batch_idx])

        #batch_idx = torch.concat(idx_dict, dim=-1)
        self.length = len(idx_dict[0])
        # for next(iter) > it only repeat the part under this comment, 
        # until iter_list ends. if it ends it goes back to top
        for i in range(self.length):
            batch_list = torch.concat((idx_dict[0][i],idx_dict[1][i],idx_dict[2][i],idx_dict[3][i]))
            batch_list = batch_list.tolist()
            shuffle(batch_list)
            yield batch_list # as it was stored in an array
            
class EmoSampler(Sampler):
    def __init__(self, data, class_type, target_class, batch_size, use_classes, equal_sampling):
        """
        target_emotion / non_target_emotion
        """
        self.data = data
        self.class_type = class_type
        self.target_class = target_class
        self.use_classes = use_classes

        self.batch_size = batch_size
        self.equal_sampling = equal_sampling

        if self.target_class not in use_classes:
            raise ValueError("input_key is not in the use_classes, check again")
        
        counter = Counter(self.data['emotion'])        
        self.n_sample = counter[self.target_class]
        if not self.equal_sampling:
            for _class in use_classes:
                if _class == self.target_class:
                    pass
                else:
                    self.data = self.data.replace(_class, 'non_' + self.target_class)
            self.use_classes = [self.target_class, 'non_' + self.target_class]
        

    def __iter__(self):
        # mask dataset first
        # print(self.data)
        # print(self.use_classes)
        
        idx_list = []
        candidate = []
        for e in self.use_classes:
            index = self.data[self.data['emotion'] == e].index.tolist()
            #print(len(index), self.n_sample)
            if self.target_class == e:
                sample = random.sample(index, self.n_sample)
            else:
                if self.equal_sampling:
                    n = self.n_sample // (len(self.use_classes) - 1)
                    n = min(n, len(index))
                    sample = random.sample(index, n)
                    if n != len(index):
                        candi = set(index) - set(sample)
                        candidate.extend(candi)
                else:
                    sample = random.sample(index, self.n_sample)
            idx_list.extend(sample)  

        left_over = 2 * self.n_sample - len(idx_list)
        if left_over > 0:
            sample = random.sample(candidate, left_over)
            idx_list.extend(sample)

        shuffle(idx_list)

        n_split = int(len(idx_list) / self.batch_size)
        batch_idx = np.array_split(idx_list, n_split)

        # for next(iter) > it only repeat this part until iter_list ends
        # if it ends it goes back to top
        for i in batch_idx: 
            yield i.tolist() # as it was stored in an array

class ClassSampler_Neu(Sampler):
    def __init__(self, data, class_list, batch_size, n_sample):
        """
        class마다 동일하게
        """
        self.data = data
        self.class_list = class_list

        self.batch_size = batch_size

        #counter = Counter(self.data['emotion'])
        
        #self.class_count = [counter[k] for k in self.class_list]
        self.min_n = n_sample #min(self.class_count)

        self.n_class = len(class_list)

    def __iter__(self):
        # mask dataset first
        idx_list = []
        #torch.manual_seed(self.seed)
        #self.seed += 1
        for e in self.class_list:
            index = self.data[self.data['emotion']==e].index.tolist()
            sample = random.sample(index, self.min_n if e != 'neu' else self.min_n * (self.n_class-1))
            
            idx_list.extend(sample)  

        shuffle(idx_list)

        to_count = self.data.iloc[idx_list]
        print(Counter(to_count['emotion']))
        
        n_split = int(len(idx_list)/self.batch_size)
        batch_idx = np.array_split(idx_list, n_split)

        self.length = len(batch_idx)

        # for next(iter) > it only repeat the part under this comment, 
        # until iter_list ends. if it ends it goes back to top
        for i in batch_idx: 
            yield i.tolist() # as it was stored in an array

class IDSampler(Sampler):
    def __init__(self, data, batch_size):
        """
        class마다 동일하게
        """
        self.data = data

        self.batch_size = batch_size
        counter = Counter(self.data['id_num'])
        self.min_n = counter.most_common()[-1][1]
        self.class_list = counter.keys()
        self.n_class = len(self.class_list)

    def __iter__(self):
        # mask dataset first
        idx_list = []
        for e in self.class_list:
            index = self.data[self.data['id_num']==e].index.tolist()
            sample = random.sample(index, self.min_n)
            idx_list.extend(sample)  

        shuffle(idx_list)

        n_split = int(len(idx_list)/self.batch_size)
        batch_idx = np.array_split(idx_list, n_split)

        self.length = len(batch_idx)

        # for next(iter) > it only repeat the part under this comment, 
        # until iter_list ends. if it ends it goes back to top
        for i in batch_idx: 
            yield i.tolist() # as it was stored in an array

class IDSampler_Strict(Sampler):
    def __init__(self, data, batch_size):
        """
        class마다 동일하게
        """
        self.data = data

        self.batch_size = batch_size
        counter = Counter(self.data['id_num'])
        self.min_n = counter.most_common()[-1][1]
        self.class_list = counter.keys()
        self.n_class = len(self.class_list)

    def __iter__(self):
        # mask dataset first
        idx_dict = []
        split_num = []
        for e in self.class_list:
            index = self.data[self.data['id_num']==e].index.tolist()
            sample = random.sample(index, self.min_n)
            shuffle(sample)

            id_n_split = int(len(sample)/(self.batch_size/len(self.class_list)))
            id_batch_idx = np.array_split(sample, id_n_split)
            idx_dict.append([torch.tensor(e_idx) for e_idx in id_batch_idx])

        #batch_idx = torch.concat(idx_dict, dim=-1)
        self.length = len(idx_dict[0])
        # for next(iter) > it only repeat the part under this comment, 
        # until iter_list ends. if it ends it goes back to top
        for i in range(self.length):
            batch_list = torch.concat([idx_dict[k][i] for k in range(len(self.class_list))])
            batch_list = batch_list.tolist()
            shuffle(batch_list)
            yield batch_list # as it was stored in an array