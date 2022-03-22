import torch
from model_MLP import MLP
from train_MLP import train_loop, init_config


train_ds = "validation"
valid_ds = "test"
bsz = 512
learning_rate = 1
MAX_epochs = 1000


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


MLPmodel = MLP(4, "sigmoid", [285,128,8,2]).to(device)
Train_loader, Valid_loader, Optim_SGD, CE_Loss = init_config(MLPmodel, train_ds, valid_ds, bsz, learning_rate, device)

train_loop(MLPmodel, Train_loader, Valid_loader, Optim_SGD, CE_Loss, MAX_epochs, device)