# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:25:07 2018

@author: dedekinds
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  8 08:47:08 2018

@author: dedekinds
"""

import numpy as np
import networkx as nx
import os
from gensim.models import Word2Vec
from sklearn.linear_model.logistic import LogisticRegression


directed = True
p = 1.0#对于node2vec中的p==q时候等价于deepwwalk
q = 1.0
num_walks = 1000
walk_length = 100
emb_size = 200
iteration = 5


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


'''
根据互引用的关系构造有向图
'''
def load_graph(filename, id_list):
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            line_split = line.split()
            #print(line_split)
        
            if line_split[0] in id_list and line_split[1] in id_list  and line_split[0] != line_split[1]:
                g.add_edge(line_split[0], line_split[1])
                g[line_split[0]][line_split[1]]['weight'] = 1
            
            line = f.readline()
    return g



def preprocess_transition_probs(g, directed = False, p=1, q=1):
    alias_nodes, alias_edges = {}, {};
    for node in g.nodes():
        probs = [g[node][nei]['weight'] for nei in sorted(g.neighbors(node))]
        norm_const = sum(probs)
        norm_probs = [float(prob)/norm_const for prob in probs]
        alias_nodes[node] = get_alias_nodes(norm_probs)

    if directed:
        for edge in g.edges():
            alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
            #print(alias_edges[edge])
    else:
        for edge in g.edges():
            alias_edges[edge] = get_alias_edges(g, edge[0], edge[1], p, q)
            alias_edges[(edge[1], edge[0])] = get_alias_edges(g, edge[1], edge[0], p, q)

    return alias_nodes, alias_edges


def get_alias_edges(g, src, dest, p=1, q=1):
    probs = [];
    for nei in sorted(g.neighbors(dest)):
        if nei==src:
            probs.append(g[dest][nei]['weight']/p)
        elif g.has_edge(nei, src):
            probs.append(g[dest][nei]['weight'])
        else:
            probs.append(g[dest][nei]['weight']/q)
    norm_probs = [float(prob)/sum(probs) for prob in probs]
    return get_alias_nodes(norm_probs)

'''
针对节点t来说，我们得到了t能转移到不同类别节点的概率，
常规做法是归一化之后按照概率随机选取，但这篇论文并没有直接这样做，而是选用了Alias算法进行抽样
'''
def get_alias_nodes(probs):
    l = len(probs)
    a, b = np.zeros(l), np.zeros(l, dtype=np.int)
    small, large = [], []

    for i, prob in enumerate(probs):
        a[i] = l*prob
        if a[i]<1.0:
            small.append(i)
        else:
            large.append(i)
            
    while small and large:
        sma, lar = small.pop(), large.pop()
        b[sma] = lar
        a[lar]+=a[sma]-1.0
        if a[lar]<1.0:
            small.append(lar)
        else:
            large.append(lar)
    return b, a


def node2vec_walk(g, start, alias_nodes, alias_edges, walk_length=30):
    path = [start]
    while len(path)<walk_length:
        node = path[-1]
        neis = sorted(g.neighbors(node))
        if len(neis)>0:
            if len(path)==1:
                l = len(alias_nodes[node][0])
                idx = int(np.floor(np.random.rand()*l))
                if np.random.rand()<alias_nodes[node][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_nodes[node][0][idx]])
            else:
                prev = path[-2]
                l = len(alias_edges[(prev, node)][0])
                idx = int(np.floor(np.random.rand()*l))
                if np.random.rand()<alias_edges[(prev, node)][1][idx]:
                    path.append(neis[idx])
                else:
                    path.append(neis[alias_edges[(prev, node)][0][idx]])
        else:
            break
    return path 







edge_path = 'data/cora/cora.content'
label_path = 'data/cora/cora.cites'
model_path = './output_deepwalk.model'

# load feature and adjacent matrix from file
id_list, labels = load_features(edge_path)
g = load_graph(label_path, id_list)#print(g)


for node in id_list:
    if not g.has_node(node):
        g.add_node(node)
        
        
if os.path.isfile(model_path):
    model = Word2Vec.load(model_path)
    print ('load model successfully')
else: 
    alias_nodes, alias_edges = preprocess_transition_probs(g, directed,p,q)


    walks = []
    idx_total = []
    for i in range(num_walks):
        r = np.array(range(len(id_list)))
        np.random.shuffle(r)
        #r = list(r)
        #idx_total+=r
        for node in [id_list[j] for j in r]:
            walks.append(node2vec_walk(g, node, alias_nodes, alias_edges, walk_length))

    model = Word2Vec(walks, size=emb_size, min_count=0, sg=1, iter=iteration)
    model.save('output_deepwalk.model')









y=[]
for temp in range(2708):
    y.append( LABEL[labels[temp]])
y = np.array(y)

x_train = np.zeros(emb_size)
x_test =  np.zeros(emb_size)


droppoint = 500

for x in range(droppoint):
    x_train = np.row_stack((x_train,model[id_list[x]]))
x_train = np.delete(x_train,[0],axis = 0)
y_train = y[:droppoint]


for x in range(droppoint,1500):
    x_test = np.row_stack((x_test,model[id_list[x]]))
x_test = np.delete(x_test,[0],axis = 0)
y_test = y[droppoint:1500]

#
#neigh = ExtraTreesClassifier()
#neigh.fit(x_train, y_train)
#preds = neigh.predict(x_test)
#print (list(preds-y_test).count(0)/500)


classifier=LogisticRegression()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)
print ('deepwalk:')
print (list(predictions-y_test).count(0)/1000)

