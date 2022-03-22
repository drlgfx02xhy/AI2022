import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MyDataset
from model_MLP import MLP
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util import setup_seed
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
import logging

def log():
    logging.basicConfig(filename='log.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                     level=logging.INFO)

def init_config(model, train_ds:str, valid_ds:str, bsz:int, learning_rate:float, device):
    log()
    setup_seed(42)
    logging.info("loading data...")
    print("loading data...")
    train_set = MyDataset(train_ds)
    valid_set = MyDataset(valid_ds)
    logging.info("loaded!")
    print("loaded!")
    train_loader = DataLoader(dataset = train_set, batch_size = bsz, shuffle=True)
    valid_loader = DataLoader(dataset = valid_set, batch_size = bsz, shuffle=True)
    Optim_SGD = torch.optim.SGD(model.parameters(), lr = learning_rate)
    CE_Loss = nn.CrossEntropyLoss().to(device)
    logging.info("initialize!")
    print("initialize!")
    # train_loop(MLPmodel, test_loader, valid_loader, Optim_SGD, CE_Loss, 10, device)
    return train_loader, valid_loader, Optim_SGD, CE_Loss

def train_loop(model, train_loader, valid_loader, optimizer, criterion, epochs, device):
    log()
    logging.info("start training!")
    print("start training!")
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            if(idx % 100 == 0):
                acuracy, F1, FPR, TPR, AUC = eval(model, valid_loader, device)
                logging.info("epoch{}, idx{}: F1:{:.5f} accs{:.5f}".format(str(epoch), str(idx), F1, acuracy))
                print("epoch{}, idx{}: F1:{:.5f} accs{:.5f}".format(str(epoch), str(idx), F1, acuracy))

def eval(model, data_loader, device):
    log()
    model.eval()
    for i, (x, y) in enumerate(data_loader):
        x = Variable(x.to(device))
        logits = F.softmax(model(x),dim = -1)
        _, y_hat = logits.topk(1, dim = -1)
        y_hat = y_hat.cpu().numpy()
        y = y.numpy()
        accs = accuracy_score(y, y_hat)
        f1 = f1_score(y, y_hat, average="binary")
        fpr, tpr, threshold = roc_curve(y, y_hat)
        scale = auc(fpr, tpr)
    return accs, f1, fpr, tpr, scale

"""  
def cal_F1(y, y_hat):
    TP = FN = FP = TN = 0
    y = y.numpy()
    y_hat = y_hat.cpu().numpy()
    # correct = (y_hat[:,-1] == y).sum()
    for i in range(len(y)):
        if(y[i] == 1 and y_hat[i] == 1):
            TP += 1
        elif(y[i] == 0 and y_hat[i] == 1):
            FP += 1
        elif(y[i] == 1 and y_hat[i] == 0):
            FN += 1
        else:
            TN += 1
    return TP,FP,FN,TN

def cal_accs(y, y_hat):
    both = len(y)
    y = y.numpy()
    y_hat = y_hat.cpu().numpy()
    correct = (y_hat[:,-1] == y).sum()
    return correct, both
"""

"""
`hp`
lr: 0.2 0.5 0.8
list: [285,2] [285,10,2] [285,64,8,2]
bsz: 128 512 1024
activate: relu tanh softmax
max_epoch = 5000
    
"""

"""train: according to F1 score

0. 3个模型: 3*3
1. 神经元数量: 3*4
2. lr: 3*3
3. 激活函数: 5
4. bsz: 5

0. svm: 5
1. soft margin: 5
2. 2 kernal: 2*5


"""

"""valid & test:

1. 4个模型在valid集的过拟合、欠拟合程度: 4*3
2. 4个模型在test集的F1 score, ROC, AUC, 是否符合valid预期，有无其他现象
    4*(1 + 2 + 1 + 2)
"""

"""
机器、语言、库
预处理和读取方式
4'
"""

        
        
        
        
        
 
    
