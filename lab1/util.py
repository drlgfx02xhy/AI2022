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
    
def get_small_data(rate, seed):
    setup_seed(seed)
    train_feat = np.load("train_feat.npy")
    train_label = np.load("train_label.npy")
    valid_feat = np.load("validation_feat.npy")
    valid_label = np.load("validation_label.npy")
    test_feat = np.load("test_feat.npy")
    test_label = np.load("test_label.npy")
    
    train_len = int(len(train_feat) * rate)
    valid_len = int(len(valid_feat) * rate)
    test_len = int(len(test_feat) * rate)
    
    select_train = random.sample(range(len(train_feat)), train_len)
    select_valid = random.sample(range(len(valid_feat)), valid_len)
    select_test = random.sample(range(len(test_feat)), test_len)
    
    new_train_feat = train_feat[select_train]
    new_train_label = train_label[select_train]
    new_valid_feat = valid_feat[select_valid]
    new_valid_label = valid_label[select_valid]
    new_test_feat = test_feat[select_test]
    new_test_label = test_label[select_test]
    
    new_train_feat_path = "small_train_feat" + ".npy"
    new_train_label_path = "small_train_label" + ".npy"
    new_valid_feat_path = "small_validation_feat" + ".npy"
    new_valid_label_path = "small_validation_label" + ".npy"
    new_test_feat_path = "small_test_feat" + ".npy"
    new_test_label_path = "small_test_label" + ".npy"
    
    np.save(new_train_feat_path, new_train_feat)
    np.save(new_train_label_path, new_train_label)
    np.save(new_valid_feat_path, new_valid_feat)
    np.save(new_valid_label_path, new_valid_label)
    np.save(new_test_feat_path, new_test_feat)
    np.save(new_test_label_path, new_test_label)
    
    return new_train_feat, new_train_label, new_valid_feat, new_valid_label, new_test_feat, new_test_label
    
    
    
    