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

setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("loading...")

# train_set = MyDataset("train")
valid_set = MyDataset("x")
test_set = MyDataset("y")

print("loading over!")

# train_loader = DataLoader(dataset = train_set, batch_size = 200, shuffle=True)
valid_loader = DataLoader(dataset = valid_set, batch_size = 20, shuffle=False)
test_loader = DataLoader(dataset = test_set, batch_size = 20, shuffle=False)

# MLPmodel = MLP(args.num_layers, args.act_f, arg.paralist, args.need_bias).to(device)
MLPmodel = MLP(3, "softmax", [285,10,2]).to(device)
print(MLPmodel)
# optimizer = torch.optim.SGD(MLPmodel.parameters(), lr = args.lr)
Optim_SGD = torch.optim.SGD(MLPmodel.parameters(), lr = 0.5)
CE_Loss = nn.CrossEntropyLoss().to(device)

def train_loop(model, train_loader, valid_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        for idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            if(idx % 200 == 0):
                print("epoch{}, idx{}: loss:{} accs: ".format(str(epoch), str(idx), str(loss.item())))
                accs = eval_loop(model, valid_loader)
                print(accs)


def eval_loop(model, data_loader):
    model.eval()
    total_correct = 0
    total_both = 0
    for i, (x, y) in enumerate(data_loader):
        x = Variable(x.to(device))
        logits = F.softmax(model(x),dim = -1)
        _, y_hat = logits.topk(1, dim = -1)
        correct, both = cal(y, y_hat)
        total_correct += correct
        total_both += both
    accs = float(total_correct / total_both)
    return accs
    
def cal(y, y_hat):
    both = len(y)
    y = y.numpy()
    y_hat = y_hat.cpu().numpy()
    correct = (y_hat[:,-1] == y).sum()
    return correct, both

train_loop(MLPmodel, test_loader, valid_loader, Optim_SGD, CE_Loss, 10, device)

"""
`hp`
lr: 0.2 0.5 0.8
list: [285,2] [285,10,2] [285,64,8,2]
bsz: 128 512 1024
activate: relu tanh softmax
max_epoch = 5000
    
"""

"""
TODO:
    设置训练提前停止（weight_decay; 10epoch不变优则停止）
    按照DL的lab1写好config文件，自动跑测试
    写好老师要求的几个准确度的衡量
    实验结果分析
"""

        
        
        
        
        
 
    
