# -*- coding: utf-8 -*-
"""
Created on Sat May 26 10:03:39 2018

@author: dedekinds
"""
import numpy as np  
import matplotlib.pyplot as plt  
from numpy import linalg as la  
from collections import Counter
from sklearn.decomposition import NMF 
from sklearn.linear_model.logistic import LogisticRegression


ft = 30#文本信息的特征,似乎太大会使得loss反弹,加大这个效果还不错
alpha = 0.000001#学习步长
beta = 0.5#惩罚系数
K = 30#图的嵌入维度
max_iter = 10
length = 2708

np.random.seed(1) 

LABEL = {
      'Case_Based':1,
		'Genetic_Algorithms':2,
		'Neural_Networks':3,
		'Probabilistic_Methods':4,
		'Reinforcement_Learning':5,
		'Rule_Learning':6,
		'Theory':7
        }

'''
得到Cora数据集中paper的ID和对应的分类label
'''
def load_features(filename):
    ids, labels = [], []
    with open(filename, 'r') as f:
        line = f.readline();
        while line:
            line_split = line.split();

            ids.append(line_split[0]);
            labels.append(line_split[-1]);
            line = f.readline();

        return ids, labels


def solve1(M,T,alpha,beta,max_iter,ft):
    n, m = M.shape  
    U = np.mat(np.random.random((n, K)))  
    V = np.mat(np.random.random((ft, K)))  
    
    _iter = 0
    loss_list = []  
    while _iter < max_iter:
        print(_iter)
        
        U = U - alpha * (2 * (U*V.transpose()*T-M)*T.transpose()*V+ beta * U)
        
        temp = T*T.transpose()*V*U.transpose()*U-T*M.transpose()*U
        V = V - alpha * (2 *temp  + beta * V)
        
        loss = np.linalg.norm(M-U*V.transpose()*T,ord=2)
        print(loss)
        loss_list.append(loss)
        if loss <= 1e-3:
            break
        _iter += 1
        
    U = np.hstack(  (U,T.transpose()*V*10)  )
    return loss_list, U, V

def solve2(M,T,alpha,beta,max_iter,ft):#等价于deepwalk
    n, m = M.shape  
    U = np.mat(np.random.random((m, K)))  
    V = np.mat(np.random.random((n, K)))  
    
    _iter = 0
    loss_list = []
    while _iter < max_iter:
        print(_iter)
        U = U - alpha * (2 * (U*V.transpose()-M)*V+ beta * U)
        V = V - alpha * (2 * (V*U.transpose()-M.transpose())*U + beta * V)
        
        loss = np.linalg.norm(M-U*V.transpose(),ord=2)
        print(loss)
        loss_list.append(loss)
        if loss <= 1e-3:
            break
        _iter += 1
    return loss_list, U, V


def solve3(M,T,alpha,beta,max_iter,ft):#非负矩阵分解
    
    U,sigma,VT=la.svd(X)  
    T = U[:length,:K]*np.diag(sigma[:K])
    for i in range(len(T)):
        T[i] = T[i]/np.linalg.norm(T[i],ord=2)
    T = T*0.1#如果不缩小的话，数值上回爆炸
    
    nmf = NMF(n_components=K , max_iter = 600 , init = 'nndsvd')  
    user_distribution = nmf.fit_transform(M)  
    item_distribution = nmf.components_
    
    for temp in range(len(user_distribution[0])):
        user_distribution[:,temp] = user_distribution[:,temp]#/10**-15
        
    user_distribution = np.hstack((user_distribution,T*10))
    
    return 0, user_distribution, item_distribution






#__________________________________________________________________
    

edge_path = 'data/cora/cora.content'
label_path = 'data/cora/cora.cites'
id_list, labels = load_features(edge_path)

#构造ID的index0-2707
ID = dict()
index = 0
for x in id_list:
    ID[x] = index
    index += 1
#计算每个ID的度

num_vex = []

with open(label_path, 'r') as f:
    line = f.readline()
    while line:
        line_split = line.split()
        num_vex.append(line_split[0])
        line = f.readline()

DU = Counter(num_vex)

#计算邻接矩阵A
A = np.zeros((length,length))
with open(label_path, 'r') as f:
    line = f.readline()
    while line:
        line_split = line.split()
        A[ID[line_split[0]]][ID[line_split[1]]] = 1/(DU[line_split[0]]+0.0001)
        
        line = f.readline()


#先对文本信息进行SVD分解

U,sigma,VT=la.svd(X)  
T = U[:length,:ft]*np.diag(sigma[:ft])
for i in range(len(T)):
    T[i] = T[i]/np.linalg.norm(T[i],ord=2)
T = T.transpose()*0.1#如果不缩小的话，数值上回爆炸
#构造新邻接矩阵M矩阵

#A = load_data()
#M=np.mat(np.random.random((length, length)))

num_walk = 50
M = A
for t in range(2,num_walk+1):
    M = M + A**t
M = M /num_walk
    
#M = (A+A*A)/2

LOSS , W ,H= solve1(M,T,alpha,beta,max_iter,ft)


#plt.plot(range(len(LOSS)), LOSS)  
#plt.show()  
#LOSS[-5:-1]



#________________________________________________
#测试
y=[]
for temp in range(2708):
    y.append( LABEL[labels[temp]])
y = np.array(y)

x_train = np.zeros(np.shape(W)[1])
x_test =  np.zeros(np.shape(W)[1])

droppoint = 200

for x in range(droppoint ):
    x_train = np.row_stack((x_train,W[ID[id_list[x]]]))
x_train = np.delete(x_train,[0],axis = 0)
y_train = y[:droppoint ]


for x in range(droppoint ,1500):
    x_test = np.row_stack((x_test,W[ID[id_list[x]]]))
x_test = np.delete(x_test,[0],axis = 0)
y_test = y[droppoint :1500]

#
#neigh = ExtraTreesClassifier()
#neigh.fit(x_train, y_train)
#preds = neigh.predict(x_test)
#print (list(preds-y_test).count(0)/500)


classifier=LogisticRegression()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)
print ('TADW:')
print (list(predictions-y_test).count(0)/1000)

