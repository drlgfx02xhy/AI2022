import pandas as pd
import numpy as np
import random
import torch
import os

train_path = r"..\data\Lab1_train.csv"
valid_path = r"..\data\Lab1_validation.csv"
test_path = r"..\data\Lab1_test.csv"

def load_data(mode):
    path = "data\Lab1_" + mode + ".csv"
    print(path)
    with open(path,encoding = 'utf-8') as f:
        content = np.loadtxt(f,str,delimiter = ",", skiprows = 1)
        feat = content[:,1:-1]
        label = content[:,-1]
    feat_path =  mode + "_feat" + ".npy"
    label_path =  mode + "_label" + ".npy"
    print("save...")
    # np.save(feat_path, feat)
    print("save...")
    # np.save(label_path, label)

# load_data("test")
# load_data("validation")
# load_data("train")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
    

