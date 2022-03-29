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

def log(idx):
    filename = "log_MLP"+ str(idx) +".txt"
    logging.basicConfig(filename = filename,
                     format = '%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                     level=logging.INFO)

def init_config(model, train_ds:str, valid_ds:str, bsz:int, learning_rate:float, device, idx):
    log(idx)
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

def train_loop(model, train_loader, valid_loader, optimizer, criterion, epochs, device, idx):
    log(idx)
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
                t_accs, t_f1, t_fpr, t_tpr, t_auc = eval(model, train_loader, device, idx)
                acuracy, F1, FPR, TPR, AUC = eval(model, valid_loader, device, idx)
                logging.info("train: epoch{}, idx{}: F1:{:.5f} accs{:.5f}".format(str(epoch), str(idx), t_f1, t_accs))
                logging.info("eval: epoch{}, idx{}: F1:{:.5f} accs{:.5f}".format(str(epoch), str(idx), F1, acuracy))
                print("epoch{}, idx{}: F1:{:.5f} accs{:.5f}".format(str(epoch), str(idx), F1, acuracy))
    return model

def eval(model, data_loader, device, idx):
    log(idx)
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
        
        
        
        
 
    
