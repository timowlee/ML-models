from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from base import BaseDataLoader
import os
import numpy as np
import torch
import pandas as pd

def balanced_by_target(speed, target, data):
    d = {'speed':speed, 'target':target,'data': data}
    d = pd.DataFrame(d)
    g = d.groupby('target')
    g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    speed = g['speed'].values.tolist()
    target = g['target'].values.tolist()
    data = g['data'].values.tolist()
    return speed, target, data

def balanced_by_speed(speed, target, data):
    d = {'speed':speed,'target':target,'data': data}
    d = pd.DataFrame(d)
    g = d.groupby('speed')
    g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    speed = g['speed'].values.tolist()
    target = g['target'].values.tolist()
    data = g['data'].values.tolist()
    return speed, target, data

def balanced_by_target_in_speed(speed, target, data):
    speed_o = []
    target_o = []
    data_o = []
    for i in set(speed):
        s_o = []
        t_o = []
        d_o = []
        for idx, j in enumerate(speed):
            if j == i:
                s_o.append(i)
                t_o.append(target[idx])
                d_o.append(data[idx])
        s_o, t_o, d_o = balanced_by_target(s_o, t_o, d_o)
        speed_o = speed_o + s_o
        target_o = target_o + t_o
        data_o = data_o + d_o
    return speed_o, target_o, data_o

def balance_by_target_and_speed(speed, target, data):
    speed_o = []
    target_o = []
    data_o = []
    for i in set(speed):
        s_o = []
        t_o = []
        d_o = []
        for idx,j in enumerate(speed):
            if j == i:
                s_o.append(i)
                t_o.append(target[idx])
                d_o.append(data[idx])
        s_o, t_o, d_o = balanced_by_target(s_o, t_o, d_o)
        speed_o = speed_o + s_o
        target_o = target_o + t_o
        data_o = data_o + d_o
        speed_o, target_o, data_o = balanced_by_speed(speed_o, target_o, data_o)
    return speed_o, target_o, data_o

class standard_dataset(Dataset):
    def __init__(self, data_dir, train_selected, data_balance, data_normalization):
        self.data_dir = data_dir
        speed = []
        data = []
        target = []
        pos_ls = []
        neg_ls = []
        for i in train_selected:
            dataset = i['dataset']
            speed_list = i['speed']
            for filename in os.listdir(data_dir):
                if filename.endswith(".npy"):
                    dataset = i['dataset'].lower()
                    speed_list = i['speed']
                    for j in speed_list:
                        if j in filename.lower() and "good" in filename.lower() and dataset in filename.lower():
                            pos_ls.append(os.path.join(data_dir, filename))
                        elif j in filename.lower() and "bad" in filename.lower() and dataset in filename.lower():
                            neg_ls.append(os.path.join(data_dir, filename))
                        else:
                            continue
                else:
                    continue
        for i in pos_ls:
            sp = i.split("Good_")[1].strip(".npy")
            s = np.load(i)
            for j in s:
                speed.append(sp)
                tmp = j.astype(np.float32)
                if data_normalization == True:
                    tmp = (tmp-min(tmp))/(max(tmp)-min(tmp))
                data.append(tmp)
                # 0: good
                target.append(0)
        for i in neg_ls:
            sp = i.split("Bad_")[1].strip(".npy")
            s = np.load(i)
            for j in s:
                speed.append(sp)
                tmp = j.astype(np.float32)
                if data_normalization == True:
                    tmp = (tmp-min(tmp))/(max(tmp)-min(tmp))
                data.append(tmp)
                # 1: bad
                target.append(1)
        if data_balance == 1:
            speed, target, data = balanced_by_target(speed, target, data)
        elif data_balance == 2:
            speed, target, data = balanced_by_speed(speed, target, data)
        elif data_balance == 3:
            speed, target, data = balanced_by_target_in_speed(speed, target, data)
        elif data_balance == 4:
            speed, target, data = balance_by_target_and_speed(speed, target, data)
        else:
            pass

        self.data = torch.tensor(data)
        self.target = torch.tensor(target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        target = self.target[idx]        
        return data, target

class standard_dataset_bk(Dataset):
    def __init__(self, data_dir, train_all, train_selected, data_balance):
        self.data_dir = data_dir
        speed = []
        data = []
        target = []
        pos_ls = []
        neg_ls = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".npy"):
                if train_all == True:
                    if "good" in filename.lower():
                        pos_ls.append(os.path.join(data_dir, filename))
                    elif "bad" in filename.lower():
                        neg_ls.append(os.path.join(data_dir, filename))
                    else:
                        continue
                else:
                    for i in train_selected:
                        if i in filename.lower() and "good" in filename.lower():
                            pos_ls.append(os.path.join(data_dir, filename))
                        elif i in filename.lower() and "bad" in filename.lower():
                            neg_ls.append(os.path.join(data_dir, filename))
                        else:
                            continue
            else:
                continue
        for i in pos_ls:
            sp = i.split("Good_")[1].strip(".npy")
            s = np.load(i)
            for j in s:
                speed.append(sp)
                data.append(j.astype(np.float32))
                # 0: good
                target.append(0)
        for i in neg_ls:
            sp = i.split("Bad_")[1].strip(".npy")
            s = np.load(i)
            for j in s:
                speed.append(sp)
                data.append(j.astype(np.float32))
                # 1: bad
                target.append(1)
        if data_balance == 1:
            speed, target, data = balanced_by_target(speed, target, data)
        elif data_balance == 2:
            speed, target, data = balanced_by_speed(speed, target, data)
        elif data_balance == 3:
            speed, target, data = balanced_by_target_in_speed(speed, target, data)
        elif data_balance == 4:
            speed, target, data = balance_by_target_and_speed(speed, target, data)
        else:
            pass
        data=np.concatenate(data,axis=0)
        self.data = torch.tensor(data)
        self.target = torch.tensor(target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        target = self.target[idx]        
        return data, target

class Standard_DataLoader(BaseDataLoader):
    def __init__(self, data_dir, train_selected, batch_size, test_selected=[], data_balance=0, data_normalization=False, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = standard_dataset(self.data_dir, train_selected, data_balance, data_normalization)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)