import numpy as np
import sklearn.svm as svm

model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',coef0=0.0, shrinking=True,probability=False,tol=0.001, 
cache_size=200,class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)
"""
kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
linear：线性分类器（C越大分类效果越好，但有可能会过拟合（default C=1））
poly：多项式分类器
rbf：高斯模型分类器（gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。）
sigmoid：sigmoid核函数
具体可以参考：https://zhuanlan.zhihu.com/p/157722218
degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’。 如果gamma是’auto’，那么实际系数是1 / n_features。
coef0 ：核函数中的独立项。 它只在’poly’和’sigmoid’中很重要。
probability ：是否启用概率估计。 必须在调用fit之前启用它，并且会减慢该方法的速度。默认为False
shrinking ：是否采用shrinking heuristic方法(收缩启发式)，默认为true
tol ：停止训练的误差值大小，默认为1e-3
cache_size ：核函数cache缓存大小，默认为200
class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
verbose ：允许冗余输出
max_iter ：最大迭代次数。-1为无限制。
decision_function_shape ：‘ovo’, ‘ovr’ or None, default=ovr
关于‘ovo’, ‘ovr’的解释：
一对多法（one-versus-rest,简称OVR SVMs）：训练时依次把某个类别的样本归为一类,其他剩余的样本归为另一类，这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。
一对一法（one-versus-one,简称OVO SVMs或者pairwise）：其做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k-1)/2个SVM。当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别。
详细讲解，可以参考这篇博客：https://blog.csdn.net/xfchen2/article/details/79621396
random_state ：数据洗牌时的种子值，int值,default=None
在随机数据混洗时使用的伪随机数生成器的种子。 如果是int，则random_state是随机数生成器使用的种子; 如果是RandomState实例，则random_state是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。
个人认为最重要的参数有：C、kernel、degree、gamma、coef0、decision_function_shape。
"""
