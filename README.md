# graph
### 关于graph可做的方向
- 图的任务也有很大的不同，可以是node-focused问题，如节点分类和链接预测，也可以是graph-focused问题，如图分类和图生成。不同的结构和任务需要不同的模型架构来处理特定的问题。   
据此，现在要做的方向是视频描述，后边可以做一些图像生成的，graph与GAN结合的一些工作，但是似乎是有了。。再具体看看。
- 关于视频描述任务：有一篇Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting,可以考虑进去，这个时域的问题;另外关于关节点做行为识别的问题，也找来看看是不是时空-图卷积

### 近期Graph的学习任务 
- semantic role labeling 代码
- Hierarchical Graph Representation Learning with Differentiable Pooling [代码](https://github.com/RexYing/diffpool)
- Learning Conditioned Graph Structures for Interpretable Visual Question Answering 代码：github.com/aimbrain/vqa-project.
condition 是基于question.那么在视频描述中也可以基于监督学习本身带有的标签，在Inference时，则，直接利用训练好的graph参数（此处可以参考一下few-shot）
- 其他关于视觉问答的论文
- Graph Neural Networks: A Review of Methods and Applications提到了一些关于视觉问答的论文可以找出来，看一看，还有关于源代码的部分
- A Comprehensive Survey on Graph Neural Networks中提到的spatial-temporal networks需要看一看，这样的网络可以结合到行为识别与视频描述任务中


### 待学习：
- 带边信息的图(Edge-informative Graph)
每条边都有信息，比如权值或边的类型。例如G2S和R-GCN。
1. **Graph-to-Sequence Learning using Gated Graph Neural Networks.**
*Daniel Beck, Gholamreza Haffari, Trevor Cohn.* ACL 2018. [paper](https://arxiv.org/pdf/1806.09835.pdf)
1. **Modeling Relational Data with Graph Convolutional Networks.**
*Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling.* ESWC 2018. [paper](https://arxiv.org/pdf/1703.06103.pdf)

- 使用不同训练方法的图变体
1. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling.**
*Jie Chen, Tengfei Ma, Cao Xiao.* ICLR 2018. [paper](https://arxiv.org/pdf/1801.10247.pdf) [[code]](https://github.com/matenure/FastGCN)



### 问题
在Graph Neural Networks: A Review of Methods and Applications论文P16中，也提到了语义角色标注这篇文章，说"special variant of the GCN "；
在语义角色标注的本文中，也提到了More formally....这一处需要再理解一些。


### 通过池化，提出了一种graph classification的方法</br>
[Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf),[code](https://github.com/RexYing/diffpool)

### 可以再根据node classification与graph classification 与edge 进行一下分类------yaya 后续任务
### 再填上 关于res-connection的相关论文


## 总结性质的
### github上的某篇总结-介绍了相关的论文、博客、以及研究者</br>
https://github.com/sungyongs/graph-based-nn</br>
https://github.com/thunlp/GNNPapers</br>
[Spatio-temporal modeling 论文列表(主要是graph convolution相关)](https://github.com/Eilene/spatio-temporal-paper-list)
### 综述论文
1. [Deep Learning on Graphs: A Survey](https://arxiv.org/abs/1812.04202) </br>
[[新智元解读]](https://mp.weixin.qq.com/s/eelcT5x_kWC0dDt0_Ph4qg)
1. [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)  </br>
[[新智元解读]](https://mp.weixin.qq.com/s/h4jQWJlQV2Ew3SpuF8k5Hw)
1. [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)</br>
1. [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261.pdf)</br>
1. [Geometric Deep Learning: Going beyond Euclidean data](https://arxiv.org/pdf/1611.08097.pdf)
1. [Computational Capabilities of Graph Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4703190)
1. [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)
1. [Non-local Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)
1. [The Graph Neural Network Model](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4700287)




## 谱上的图卷积发展:spectral-based graph convolutional networks

- 以下四篇是按照时间轴，依次在前一篇的文章上进行改进的
1. [The Emerging Field of Signal Processing on Graphs](https://arxiv.org/pdf/1211.0053.pdf)
1. [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)
1. [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), [[PyTorch Code]](https://github.com/xbresson/graph_convnets_pytorch/blob/master/README.md) [[TF Code]](https://github.com/mdeff/cnn_graph)
1. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), [[Code]](https://github.com/tkipf/gcn), [[Blog]](http://tkipf.github.io/graph-convolutional-networks/)

- 以下三篇是在"A Comprehensive Survey on Graph Neural Networks"这篇综述中提到的另外三篇
1. [Deep convolutional networks on graph-structured data](https://arxiv.org/abs/1506.05163)
1. [Adaptive graph convolutional neural networks](https://arxiv.org/abs/1801.03226) (AAAI 2018) 可接受任意图结构和规模的图作为输入
1. [Cayleynets: Graph convolutional neural networks with complex rational spectral filters](https://arxiv.org/abs/1705.07664)


- **谱上的图卷积网络的缺陷：** (by "A Comprehensive Survey on Graph Neural Networks)   
**spectral methods usually handle the whole graph simultaneously and are difficult to parallel or scale to large graphs** (P2)</br>
**more drawbacks in "A Comprehensive Survey on Graph Neural Networks"** (P7 4.1.3 summary of spectral methods)</br>
1. 任何对graph的扰动都可以导致特征基U(特征向量)的扰动
1. 可学习的filter是与domain相关的，不能应用于不同的graph structure
1. 特征值分解需要很大的计算量和存储量
1. 虽然ChebNet and 1stChebNet定义的过滤器在空间上的局部的，且在graph上的任意位置(node)是共享的，但是这两个模型都需要载入整个graph进行graph convolution的计算，在处理big graph上计算效率低：***by yaya: X'=AXW, X'的更新, 需要输入整个X才可以计算得到***


## 空间上的图卷积：spatial-based graph convolutional networks
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. [Inductive representation learning on large graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)  (GraphSAGE)</br>
Instead of updating states over all nodes, GraphSage proposes a batch-training algorithm(sub-graph training)which improves scalability for large graphs. The learning process: P9 in "A Comprehensive Survey on Graph Neural Networks"
1. [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)  (MPNNs)
1. [Learning convolutional neural networks for graphs](https://arxiv.org/abs/1605.05273)  (PATCHY-SAN)
1. [Geometric deep learning on graphs and manifolds using mixture model cnns](http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf)
1. [Learning convolutional neural networks for graphs](http://proceedings.mlr.press/v48/niepert16.pdf)
1. [Large-scale learnable graph convolutional networks](https://dl.acm.org/citation.cfm?id=3219947)  (LGCN) 
1. [Diffusion-convolutional neural networks](https://arxiv.org/abs/1511.02136) (NIPS 2016)
1. [Geometric deep learning on graphs and manifolds using mixture model cnns](https://arxiv.org/abs/1611.08402) (CVPR 2017)
1. etc: by "A Comprehensive Survey on Graph Neural Networks" P5;P7的表格分别列举了一些spatial-based GCN 

- **Together with sampling strategies, the computation can be performed in a batch of nodes instead of the whole graph** (GraphSAGE and LGCN)


## 谱上与空间GCN的比较：Comparison Between Spectral and Spatial Models
- **by "A Comprehensive Survey on Graph Neural Networks"**
- **bridges:** The graph convolution defined by 1stChebNet(semi-supervised GCN) is localized in space. It bridges the gap between spectral-based methods and spatial-based methods. --P2
- **Drawbacks** to spectralbased models. We illustrate this in the following from three aspects, efficiency, generality and flexibility. --P11 

1. **Efficiency**</br>
**基于谱的模型**或者需要计算特征向量，或者需要同时处理整个graph，这样的情况下，模型的计算量将随着graph size 显著的增加</br>
**基于空间的模型**通过聚合临近节点的特征，直接在graph domain进行卷积计算，因此具有处理large graph的潜力。另外，可以以批次处理节点，而不是整个graph。再另外，随着临近节点的增加，可以使用采样策略来提高效率----参见后文 [改善GCN在训练方面的缺陷: Training Methods](#改善gcn在训练方面的缺陷-training-methods)</br>
1. **Generality**</br>
**基于谱的模型**假设在固定的graph上进行训练，很难泛化到其他的新的或者不同的graph上</br>
**基于空间的模型**在每个node上执行graph convolution计算，因此训练得到的权重(weights)可以轻易的共享到其他的node或者graph</br>
1. **Flexibility**</br>
**基于谱的模型**受限于无向图，但是却没有在有向图上的关于拉普拉斯矩阵(Laplacian matrix)清晰的定义。因此，若将基于谱的方法应用在有向图上，需要将有向图转化为无向图</br>
**基于空间的模型**处理多源输入更加灵活，这里的多源输入可以指：edge features or edge directions, etc</br>
**关于edge features**, 参见下文 [输入含有边特征的GNN：input allow edge features](#输入含有边特征的gnninput-allow-edge-features)</br>


## 改善GCN在训练方面的缺陷: Training Methods
- **by "A Comprehensive Survey on Graph Neural Networks"**
- [1stChebNet(semi-supervised GCN)](https://arxiv.org/abs/1609.02907)：the main drawback of 1stChebNet is that the computation cost increases exponentially with the increase of the number of 1stChebNet layers during batch training. Each node in the last layer has to expand its neighborhood recursively across previous layers.
1. [Fastgcn: fast learning with graph convolutional networks via importance sampling (ICLR 2018)](https://arxiv.org/abs/1801.10247)</br> 
assume the rescaled adjacent matrix A comes from a sampling distribution. 
1. [Stochastic training of graph convolutional networks with variance reduction (ICML 2018)](https://arxiv.org/abs/1710.10568)</br> 
reduce the receptive field size of the graph convolution to an arbitrary small scale by sampling neighborhoods and using historical hidden representations.
1. [Adaptive sampling towards fast graph representation learning (NIPS 2018)](https://arxiv.org/abs/1809.05343) </br>
propose an adaptive layer-wise sampling approach to accelerate the training of 1stChebNet, where sampling for the lower layer is conditioned on the top one. 

- **by "Graph Neural Networks: A Review of Methods and Applications"**
- GCN requires the full graph Laplacian, which is computational-consuming for large graphs. Furthermore, The embedding of a node at layer L is computed recursively by the embeddings of all its neighbors at layer L − 1. Therefore, the receptive field of a single node grows exponentially with respect to the number of layers, so computing gradient for a single node costs a lot. Finally, GCN is trained
independently for a fixed graph, which lacks the ability for inductive learning.
1. [Inductive representation learning on large graphs (NIPS 2017)](https://arxiv.org/abs/1706.02216)</br>
1. [Fastgcn: fast learning with graph convolutional networks via importance sampling (ICLR 2018)](https://arxiv.org/abs/1801.10247)</br>
directly samples the receptive field for each layer.
1. [Adaptive sampling towards fast graph representation learning (NIPS 2018)](https://arxiv.org/abs/1809.05343) </br>
 introduces a parameterized and trainable sampler to perform layerwise sampling conditioned on the former layer.
1. [Stochastic training of graph convolutional networks with variance reduction (ICML 2018)](https://arxiv.org/abs/1710.10568)</br>
proposed a control-variate based stochastic approximation algorithms for GCN by utilizing the historical activations of nodes as a control variate. 
1. [Deeper insights into graph convolutional networks for semi-supervised learning  (arXiv:1801.07606, 2018)](https://arxiv.org/abs/1801.07606)</br>  



## Graph Attention Networks
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. [Graph Attention Network (GAT)](https://arxiv.org/abs/1710.10903)(ICLR 2017)
1. [Gaan:Gated attention networks for learning on large and spatiotemporal graphs (GAAN)]
1. [Graph classification using structural attention](http://ryanrossi.com/pubs/KDD18-graph-attention-model.pdf)(ACM SIGKDD 2018)
1. [Watch your step: Learning node embeddings via graph attention]()(NIPS 2018)


## Graph Auto-encoders
<div align=center><img width="400" height="200" src="https://github.com/ShiYaya/graph/blob/master/images/graph-auto-encoder.png"/></div>

- **by "A Comprehensive Survey on Graph Neural Networks"**</br>
- network embedding算法可以分类为:1.matrix factorization 2.random walks 3. deep learning. Graph Auto-encoders是deep learning的一类方法. P2</br>
-  Network embedding是为了将node embedding 转化到低维的向量空间，通过保存网络的拓扑结构与节点内容信息，接下来的graph分析任务(比如，分类，聚类和推荐等)可以被应用于现有的机器学习任务(如SVM for classification)</br>

1. [Variational graph auto-encoders (GAE)](https://arxiv.org/abs/1611.07308) [[code]](https://github.com/tkipf/gae)</br>
 used in link prediction task in citation networks</br>
 encoder对node embedding进行更新,decoder对A(adjacency matrix)进行更新
1. [Adversarially regularized graph autoencoder for graph embedding (ARGA)](https://arxiv.org/abs/1611.07308)
1. [Learning deep network representations with adversarially regularized autoencoders (NetRA)](http://www.cs.ucsb.edu/~bzong/doc/kdd-18.pdf)
1. [Deep neural networks for learning graph representations (DNGR)](https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf)
1. [Structural deep network embedding (SDNE)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)
1. [Deep recursive network embedding with regular equivalence (DRNE)](http://pengcui.thumedialab.com/papers/NE-RegularEquivalence.pdf)



## Graph Generative Networks
- **by "A Comprehensive Survey on Graph Neural Networks"**</br>
- factor the generation process as forming nodes and edges alternatively
1. [Graphrnn: A deep generative model for graphs](https://arxiv.org/abs/1802.08773) (ICML2018)
1. [Learning deep generative models of graphs](https://arxiv.org/abs/1803.03324) (ICML2018)
- employ generative adversarial training
1. [Molgan: An implicit generative model for small molecular graphs](https://arxiv.org/pdf/1805.11973.pdf) (arXiv:1805.11973 2018)
1. [Net-gan: Generating graphs via random walks](https://arxiv.org/abs/1803.00816) (ICML2018)


## GCN Based Graph Spatial-Temporal Networks
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. [Diffusion convolutional recurrent neural network: Data-driven traffic forecasting (DCRNN)](https://arxiv.org/abs/1707.01926) (ICLR 2018)
1. [Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting (CNN-GNN)](https://arxiv.org/abs/1709.04875) (IJCAI 2017)
1. [Spatial temporal graph convolutional networks for skeleton-based action recognition (ST-GCN)](https://arxiv.org/abs/1801.07455) (AAAI 2018)
1. [Structural-rnn:Deep learning on spatio-temporal graphs (Structural-RNN)](https://arxiv.org/abs/1511.05298)  (CVPR 2016)

- **by yaya**
- 这两篇都是skeleton-based action recognition
1. [Skeleton-Based Action Recognition with Spatial Reasoning and Temporal Stack Learning] (ECCV 2018)
1. [Spatial temporal graph convolutional networks for skeleton-based action recognition (ST-GCN)] (AAAI 2018)


## 输入含有边特征的GNN：input allow edge features 
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. GNN (2009) The graph neural network model
1. MPNN (2017) Neural message passing for quantum chemistry
1. DCNN (2016) Diffusion-convolutional neural networks
1. PATCHY-SAN (2016) Learning convolutional neural networks for graphs

- **by "Deep Learning on Graphs: A Survey"**
1. Geniepath:Graph neural networks with adaptive receptive paths
1. Dual graph convolutional networks for graph-based semi-supervised classification
1. Signed graph convolutional network

- **by yaya**
1. [Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/abs/1703.04826)</br>
1. [Exploring Visual Relationship for Image Captioning](https://arxiv.org/abs/1809.07041)


## 图表达：Graph level representation/Readout Operations
**Order invariance**  A critical requirement for the graph readout operation is that the operation should be invariant to the order
of nodes, i.e. if we change the indices of nodes and edges using a bijective function between two vertex sets, representation of the whole graph should not change.   

**一. Statistics** </br>
- **by "Deep Learning on Graphs: A Survey"**
- The most basic operations that are order invariant are simple statistics like taking **sum**, **average** or **max-pooling**
1. Convolutional networks on graphs for learning molecular fingerprints
1. Diffusion-convolutional neural networks
- other 
1. Molecular graph convolutions: moving beyond fingerprints
1. Spectral networks and locally connected networks on graphs

**二. Hierarchical clustering** </br>
- **by "Deep Learning on Graphs: A Survey"**
1. Spectral networks and locally connected networks on graphs
1. Deep convolutional networks on graph-structured data
1. [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf) [[code]](https://github.com/RexYing/diffpool)

**三. Graph Pooling Modules** 
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. Convolutional neural networks on graphs with fast localized spectral filtering (NIPS 2016)
1. Deep convolutional networks on graph-structured data
1. An end-to-end deep learning architecture for graph classification (AAAI 2018) [[code]](https://github.com/muhanzhang/DGCNN)  [[pytorch code]](https://github.com/muhanzhang/pytorch_DGCNN)
1. Hierarchical graph representation learning with differentiable pooling (NIPS 2018) [[code]](https://github.com/RexYing/diffpool)

# GCN的应用

## 自然语言处理
- **by "Deep Learning on Graphs: A Survey"**

1. [Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/abs/1703.04826)</br>
* [[官方code(theano 0.8.2,lasagne 0.1)]](https://github.com/diegma/neural-dep-srl)  [[复现pytorch]](https://github.com/kervyRivas/Graph-convolutional)
* [专知讲解](https://mp.weixin.qq.com/s/c6ZhSk4r3pvnjHsvpwkkSw)
* by yaya:阅读该篇文章主要是来源于这篇将图卷积用于图像描述的文章:Exploring Visual Relationship for Image Captioning</br>
这两篇文章采用的图卷积公式都是一样的，但是我认为很奇怪，而且b是如何由edge获得的，将进一步阅读代码，稍后解释。</br>
<div align=center><img src="https://github.com/ShiYaya/graph/blob/master/images/gcn%2Bformulation.png"/></div>

1. [Graph Convolutional Encoders for Syntax-aware Neural Machine Translation](https://arxiv.org/pdf/1704.04675)


### 视觉问答( by "Graph Neural Networks: A Review of Methods and Applications")
1. **A simple neural network module for relational reasoning.**
*Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap.* NIPS 2017. [paper](https://arxiv.org/pdf/1706.01427.pdf)
1. **Graph-Structured Representations for Visual Question Answering.**
*Damien Teney, Lingqiao Liu, Anton van den Hengel.* CVPR 2017. [paper](https://arxiv.org/pdf/1609.05600.pdf)
1. **Out of the Box: Reasoning with Graph Convolution Nets for Factual Visual Question Answering.**
*Medhini Narasimhan, Svetlana Lazebnik, Alexander Schwing.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7531-out-of-the-box-reasoning-with-graph-convolution-nets-for-factual-visual-question-answering.pdf)
1. **Learning Conditioned Graph Structures for Interpretable Visual Question Answering.**
*Will Norcliffe-Brown, Efstathios Vafeias, Sarah Parisot.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1806.07243)  [[code]](https://github.com/aimbrain/vqa-project)


## **library**

3. https://github.com/rusty1s/pytorch geometric
4. https://www.dgl.ai/

- **[geometric learning library](https://github.com/rusty1s/pytorch_geometric)** 
in PyTorch named PyTorch Geometric, which implements serveral graph neural networks including ChebNet, 1stChebNet, GraphSage, MPNNs, GAT and SplineCNN.
- **[Deep Graph Library (DGL)](https://www.dgl.ai/)**
provides a fast implementation of many graph neural networks with a set of functions on top of popular deep learning platforms such as PyTorch and MXNet
