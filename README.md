# Graph-Embedding
These are the homework of Advance Machine Learning course in Tsinghua University(2018spring). Here, I implement four network embedding algorithms(deepwalk, node2vec, TADW ,LINE) for two datasets(Cora, Tencent Weibo).


# dataset
* Cora (citation dataset)
* Tencent Weibo (following network)

# Usage
If you can not run the code through the following commad, you can use python IDE(such as Spyder) to run my code.
## deepwalk
```
python hw2_deepwalk.py
```
## node2vec
```
python hw2_node2vec.py
```
## LINE
```
python hw2_deepwalk.py
```
## TADW
```
python data_utils_cora.py
python hw2_TADW.py
```

# Reference
* DeepWalk: Online Learning of Social Representations. Bryan Perozzi, Rami Al-Rfou, Steven Skiena. KDD 2014. 
* node2vec: Scalable Feature Learning for Networks. Aditya Grover, Jure Leskovec. KDD 2016.
* LINE: Large-scale Information Network Embedding. Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Me. WWW 2015.
* Network Representation Learning with Rich Text Information. Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, Edward Y. Chang. IJCAI 2015. 
* [graph_attack](https://github.com/kartikpalani/graph_attack)
* [LINE](https://github.com/VahidooX/LINE)
