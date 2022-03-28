import torch
from model_MLP import MLP
from train_MLP import train_loop, init_config
import os
import logging

logging.basicConfig(filename='log_MLP3.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                     level=logging.INFO)

idx = 3

train_ds = "train"
valid_ds = "validation"
bsz = 512
learning_rate = 0.3
MAX_epochs = 5
activator = "sigmoid"
N_layer = 4
layer_list = [285,128,16,2]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

logging.info("batch_size: {}, learning_rate: {:.2f}, max_epoch: {}, activator: {}, layer_list: {}".format(str(bsz), learning_rate, str(MAX_epochs), str(activator), layer_list))

MLPmodel = MLP(N_layer, activator, layer_list).to(device)
Train_loader, Valid_loader, Optim_SGD, CE_Loss = init_config(MLPmodel, train_ds, valid_ds, bsz, learning_rate, device, idx)

model = train_loop(MLPmodel, Train_loader, Valid_loader, Optim_SGD, CE_Loss, MAX_epochs, device, idx)
# save model
path = "./model_MLP3_2.pt"
torch.save(model.state_dict(), path)