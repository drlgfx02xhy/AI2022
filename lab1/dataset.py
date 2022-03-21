import torch
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, mode):
        super(MyDataset, self).__init__()
        feat_path = mode + "_feat.npy"
        label_path = mode + "_label.npy"
        feat = np.load(feat_path).astype(np.float)
        label = np.load(label_path).astype(np.float)
        self.feat = torch.from_numpy(feat)
        self.label = torch.from_numpy(label).long()
        
    def __getitem__(self, index):
        feat_ = self.feat[index]
        label_ = self.label[index]
        
        return feat_, label_
    
    def __len__(self):
        return len(self.label)
        
    