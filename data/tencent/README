This directory contains the a selection of the Tencent Weibo dataset [1].

This dataset consists of 169,209 users and 500,000 "following" relationship. We divide it into 6 files.

- adj_train.npz: sparse adjacency matrix of graph. The weight associated with each directed edge is 1.0. You can use it for training and load graph via load_npz function in scipy.sparse package.
- train_edges.npy: training edges in graph, in #n_train * 2 numpy array. Each row represents two node id with directed edges. You can load edges via load function in numpy. The following files have the same format as this file.
- val_edges.npy: positive validation edges in graph, in #n_valid * 2 numpy array. We select 5% edges for validation.
- val_edges_false.npy: negative validation edges in graph, in #n_valid_neg * 2 numpy array. The negative edges do not exist in graph.
- test_edges.npy: positive test edges in graph, in #n_test * 2 numpy array. We select 10% edges for test.
- test_edges_false.npy: negative test edges in graph, in #n_test_neg * 2 numpy array. The negative edges do not exist in graph.


[1] Jing Zhang, Jie Tang, Cong Ma, Hanghang Tong, Yu Jing, and Juanzi Li. Panther: Fast Top-k Similarity Search on Large Networks. In Proceedings of the Twenty-First ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'15), pages 1445-1454.