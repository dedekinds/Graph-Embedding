
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:25:21 2018

@author: dedekinds
主要参考：https://github.com/VahidooX/LINE
"""

#!/usr/bin/env python
# coding=utf-8

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.layers import Embedding, Reshape, Merge, Activation, Input, merge
from keras.models import Sequential, Model
import keras.backend as K
import warnings
warnings.filterwarnings("ignore")

auc_thd = 0.01 #设置auc的一个阈值

def LINE(train_edges,test_edges,test_edges_false):
    epoch_num = 10
    factors = 100
    batch_size = 1000
    negative_sampling = "UNIFORM" #针对负采样
    negativeRatio = 2
    
    np.random.seed(2018)
    adj_list=train_edges
    epoch_train_size=100
    print('epoch_train_size: ',epoch_train_size)
    numNodes = np.max(adj_list.ravel()) + 1
    data_gen = batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling)
    #data_gen:---左节点~右节点~是否连有边

    model_used, embed_generator = create_model(numNodes, factors)
    #model_used: 编码后的节点的乘积
    #embed_generator: 节点的编码函数
    model_used.compile(optimizer='rmsprop', loss={'left_right_dot': LINE_loss})#编译
    
    model_used.fit_generator(data_gen, samples_per_epoch=epoch_train_size, nb_epoch=epoch_num, verbose=1)
    #此时已经完成了所有节点的编码工作

#————————————————————————————————————————————————————————————————————————————————————

    
    #利用test集来输入进编码的函数得到分类器的测试集
    data_to_test=np.concatenate((test_edges,test_edges_false),axis=0)
    data_label_to_test=np.zeros(data_to_test.shape[0])
    bias=test_edges.shape[0]
    for i in range(bias):
        data_label_to_test[i]=1
        

    total_len=data_to_test.shape[0]
    #用内积表示相似度
    pred = []
    for i in range(total_len):
        print(i)
        x=embed_generator.predict_on_batch([np.asarray([data_to_test[i,0]]),np.asarray([data_to_test[i,1]])])
        dot = np.dot(x[0][0],x[1][0])
        if dot > auc_thd:
            pred.append(1)
        else:
            pred.append(0)
    
    pred=np.array(pred)
    print('LINE:')
    print('auc: ',accuracy_score(pred,data_label_to_test))




def create_model(numNodes, factors):

    left_input = Input(shape=(1,))
    right_input = Input(shape=(1,))

    left_model = Sequential()
    left_model.add(Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False))
    left_model.add(Reshape((factors,)))

    right_model = Sequential()
    right_model.add(Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False))
    right_model.add(Reshape((factors,)))

    left_embed = left_model(left_input)
    right_embed = left_model(right_input)

    left_right_dot = merge([left_embed, right_embed], mode="dot", dot_axes=1, name="left_right_dot")

    model = Model(input=[left_input, right_input], output=[left_right_dot])
    embed_generator = Model(input=[left_input, right_input], output=[left_embed, right_embed])
    #model: 编码后的节点的乘积
    #embed_generator: 节点的编码函数
    return model, embed_generator




def LINE_loss(y_true, y_pred):
    coeff = y_true*2 - 1
    return -K.mean(K.log(K.sigmoid(coeff*y_pred)))


def batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling):


    batch_size_ones = np.ones((batch_size), dtype=np.int8)
    nb_train_sample = adj_list.shape[0]
    index_array = np.arange(nb_train_sample)

    nb_batch = int(np.ceil(nb_train_sample / float(batch_size)))
    #batch分组 ，取上取整
    batches = [(i * batch_size, min(nb_train_sample, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    #确定batch分组的序号

    while 1:
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            pos_edge_list = index_array[batch_start:batch_end]
            pos_left_nodes = adj_list[pos_edge_list, 0]
            pos_right_nodes = adj_list[pos_edge_list, 1]

            pos_relation_y = batch_size_ones[0:len(pos_edge_list)]

            neg_left_nodes = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int32)
            neg_right_nodes = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int32)

            neg_relation_y = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int8)
            #构造负采样的边和节点


            left_nodes = np.concatenate((pos_left_nodes, neg_left_nodes), axis=0)
            right_nodes = np.concatenate((pos_right_nodes, neg_right_nodes), axis=0)
            relation_y = np.concatenate((pos_relation_y, neg_relation_y), axis=0)

            yield ([left_nodes, right_nodes], [relation_y])


if __name__=='__main__':
    adj_train_path='./data/tencent/adj_train.npz'
    train_edges_npy='./data/tencent/train_edges.npy'
    test_edges_npy='./data/tencent/test_edges.npy'
    test_edges_false_npy='./data/tencent/test_edges_false.npy'


#——————————————————————————————————————————————————————————————
    #读取数据
    train_edges=np.load(train_edges_npy)
    test_edges=np.load(test_edges_npy)
    test_edges_false=np.load(test_edges_false_npy)

#——————————————————————————————————————————————————————————————
    LINE(train_edges,test_edges,test_edges_false)


