import numpy as np
import sklearn.svm as svm
import torch
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
import logging
import joblib

model = svm.SVC(C=1.0, kernel='rbf')
"""
kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
linear：线性分类器（C越大分类效果越好，但有可能会过拟合（default C=1））
poly：多项式分类器
rbf：高斯模型分类器（gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。）
"""

def log():
    filename = "log_SVM.txt"
    logging.basicConfig(filename = filename,
                     format = '%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                     level=logging.INFO)

log()

# load train and validation data
print("Loading data...")
logging.info("Loading data...")
train_feat = np.load('/home/mist/xhyAIlab/lab1/train_feat.npy')
train_label = np.load('/home/mist/xhyAIlab/lab1/train_label.npy')
train_label = np.array(train_label, dtype=np.int)
valid_feat = np.load('/home/mist/xhyAIlab/lab1/validation_feat.npy')
valid_label = np.load('/home/mist/xhyAIlab/lab1/validation_label.npy')
valid_label = np.array(valid_label, dtype=np.int)

print("Data loaded.")
logging.info("Data loaded.")

print("Training SVM...")
logging.info("Training SVM...")
model.fit(train_feat, train_label)

print("SVM inference...")
logging.info("SVM inference...")
t_label = np.array(model.predict(train_feat),dtype=np.int)
pre_label = np.array(model.predict(valid_feat),dtype=np.int)

t_accs = accuracy_score(train_label, t_label)
accs = accuracy_score(valid_label, pre_label)
t_f1 = f1_score(train_label, t_label, average='binary')
f1 = f1_score(valid_label, pre_label, average="binary")
# fpr, tpr, threshold = roc_curve(y, y_hat)
# scale = auc(fpr, tpr)


logging.info("train: F1:{:.5f} accs{:.5f}".format(t_f1, t_accs))
logging.info("eval: F1:{:.5f} accs{:.5f}".format(f1, accs))

print("train: F1:{:.5f} accs{:.5f}".format(t_f1, t_accs))
print("eval: F1:{:.5f} accs{:.5f}".format(f1, accs))

# save model
path = "./model_SVM_1.pkl"
joblib.dump(model, path)

