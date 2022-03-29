import numpy as np
import torch
import joblib
import os
from model_MLP import MLP
from sklearn.metrics import roc_curve, auc, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging

def getlog(mode):
    logging.basicConfig(filename='log_test_' + mode + '.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                     level=logging.INFO)

def get_data(mode, small=False):
    print("get data")
    logging.info("get data")
    feat_path = "./"+ mode +"_feat.npy" if small==False else "./small"+ mode +"_feat.npy"
    label_path = "./"+ mode +"_label.npy" if small==False else "./small"+ mode +"_label.npy"
    feat = np.load(feat_path)
    feat = np.array(feat, dtype=np.float32)
    label = np.load(label_path)
    label = np.array(label, dtype=int)
    print("load data done")
    logging.info("load data done\n")
    return feat, label

def init_MLP(num_layers):
    # load torch model
    print("load torch model")
    logging.info("load torch model")
    if(num_layers == 2):
        mlp = MLP(2, "sigmoid", [285,32,2])
        mlp.load_state_dict(torch.load("./model_MLP_2_2.pt"))
    elif(num_layers == 3):
        mlp = MLP(3, "sigmoid", [285,128,16,2])
        mlp.load_state_dict(torch.load("./model_MLP_3_2.pt"))
    elif(num_layers == 4):
        mlp = MLP(4, "sigmoid", [285,64,16,4,2])
        mlp.load_state_dict(torch.load("./model_MLP_4_2.pt"))
    mlp.cuda()
    print("load torch model done")
    logging.info("load torch model done\n")
    return mlp

def init_SVM():
    print("load SVM model")
    logging.info("load SVM model")
    svm = joblib.load("./model_SVM_100.0_rbf.pkl")
    print("load SVM model done")
    logging.info("load SVM model done\n")
    return svm

def predict(model, mode, test_feat, test_label, train_feat, train_label):
    assert mode in ["MLP", "SVM"]
    # F1 score, ROC, AUC
    if(mode == "MLP"):
        test_feat = torch.from_numpy(test_feat).float().cuda()
        print("predict MLP")
        logging.info("predict MLP")
        logits = F.softmax(model(test_feat),dim = -1)
        train_logits = F.softmax(model(torch.from_numpy(train_feat).float().cuda()),dim = -1)
        pred_label = torch.argmax(logits, dim = 1).cpu().numpy()
        train_pred_label = torch.argmax(train_logits, dim = 1).cpu().numpy()
        
    elif(mode == "SVM"):
        print("predict SVM")
        logging.info("predict SVM")
        pred_label = model.predict((test_feat))
        train_pred_label = model.predict((train_feat))
        
    F1_score, fpr, tpr, roc_auc = cal(test_label, pred_label)
    train_F1_score = f1_score(train_label, train_pred_label, average='binary')
    draw(fpr, tpr, roc_auc)
        
    print("predict SVM done")
    logging.info("predict SVM done\n")
    return F1_score, train_F1_score
        

def cal(y, pred_y):
    F1_score = f1_score(y, pred_y, average='binary')
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    roc_auc = auc(fpr, tpr)
    return F1_score, fpr, tpr, roc_auc
    
def draw(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_curve')
    plt.legend(loc="lower right")
    plt.show()
    
def TEST(mode, layers = 0):
    getlog(mode)
    if(mode == "MLP"):
        train_feat, train_label = get_data("train", small=False)
        test_feat, test_label = get_data("test", small=False)
        model = init_MLP(layers)
    if(mode == "SVM"):
        feat, label = get_data("train", small=True)
        test_feat, test_label = get_data("test", small=True)
        model = init_SVM()

    F1_score, train_F1_score = predict(model, type, test_feat, test_label, train_feat, train_label)
    print("F1_score:{}, train_F1_score:{}".format(F1_score, train_F1_score))
    logging.info("F1_score:{}, train_F1_score:{}\n\n".format(F1_score, train_F1_score))
    
def main():
    TEST(mode="MLP", layers = 2)
    TEST(mode="MLP", layers = 3)
    TEST(mode="MLP", layers = 4)
    TEST("SVM")
    
if __name__ == "__main__":
    main()