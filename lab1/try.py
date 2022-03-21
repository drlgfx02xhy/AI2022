import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# a = torch.tensor([[1.,2.,3.],[9.,3.,4.]]).cuda()
# logits = F.softmax(a,-1)
# logits = logits.unsqueeze(0)
# x, y = logits.topk(1,-1)
# g = torch.tensor([2,0]).cuda()
# print(type(len(g)))
# print(torch.cuda.is_available())
# print(float(3/2))

# a = np.load(r"D:\USTC\two_down\AI\lab\lab1\test_feat.npy")
# feat = a[0:1]
# print(feat)
# b = np.load(r"D:\USTC\two_down\AI\lab\lab1\test_label.npy")
# label = b[0:10000]
# print(label.astype(float).sum())


# np.save(r"D:\USTC\two_down\AI\lab\lab1\x_feat.npy", feat)
# np.save(r"D:\USTC\two_down\AI\lab\lab1\x_label.npy", label)
a =[285,100,2]
i=0
for i, _ in enumerate(a[:-2]):
    print(_)
print(a[i+1])
    
    