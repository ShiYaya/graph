# graph


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




## 谱上的图卷积发展:Spectral-based graph convolutional networks

- 以下四篇是按照时间轴，依次在前一篇的文章上进行改进的
1. [The Emerging Field of Signal Processing on Graphs](https://arxiv.org/pdf/1211.0053.pdf)
1. [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)
1. [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), [[PyTorch Code]](https://github.com/xbresson/graph_convnets_pytorch/blob/master/README.md) [[TF Code]](https://github.com/mdeff/cnn_graph)
1. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), [[Code]](https://github.com/tkipf/gcn), [[Blog]](http://tkipf.github.io/graph-convolutional-networks/)

- 以下三篇是在"A Comprehensive Survey on Graph Neural Networks"这篇综述中提到的另外三篇
1. [Deep convolutional networks on graph-structured data](https://arxiv.org/abs/1506.05163)
1. [Adaptive graph convolutional neural networks](https://arxiv.org/abs/1801.03226) (AAAI 2018) 可接受任意图结构和规模的图作为输入
1. [Cayleynets: Graph convolutional neural networks with complex rational spectral filters](https://arxiv.org/abs/1705.07664)


- **谱上的图卷积网络的缺陷：** 
- **by "A Comprehensive Survey on Graph Neural Networks**
- **spectral methods usually handle the whole graph simultaneously and are difficult to parallel or scale to large graphs** (P2)</br>
- **more drawbacks:** (P7 4.1.3 summary of spectral methods)</br>
1. 任何对graph的扰动都可以导致特征基U(特征向量)的扰动
1. 可学习的filter是与domain相关的，不能应用于不同的graph structure
1. 特征值分解需要很大的计算量和存储量
1. 虽然ChebNet and 1stChebNet定义的过滤器在空间上的局部的，且在graph上的任意位置(node)是共享的，但是这两个模型都需要载入整个graph进行graph convolution的计算，在处理big graph上计算效率低：***by yaya: X'=AXW, X'的更新, 需要输入整个X才可以计算得到***


## 空间上的图卷积：Spatial-based graph convolutional networks
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. [Inductive representation learning on large graphs(GraphSAGE)](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf) [[tf code]](https://github.com/williamleif/GraphSAGE) </br>
Instead of updating states over all nodes, GraphSage proposes a batch-training algorithm(sub-graph training)which improves scalability for large graphs. The learning process: P9 in "A Comprehensive Survey on Graph Neural Networks"
1. [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf)  (MPNNs)
1. [Learning convolutional neural networks for graphs](https://arxiv.org/abs/1605.05273)  (PATCHY-SAN)
1. [Geometric deep learning on graphs and manifolds using mixture model cnns](http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf)
1. [Learning convolutional neural networks for graphs](http://proceedings.mlr.press/v48/niepert16.pdf)
1. [Large-scale learnable graph convolutional networks (LGCN)](https://dl.acm.org/citation.cfm?id=3219947) [[tf code]](https://github.com/divelab/lgcn/)
1. [Diffusion-convolutional neural networks](https://arxiv.org/abs/1511.02136) (NIPS 2016) [[tf code]](https://github.com/liyaguang/DCRNN)
1. [Geometric deep learning on graphs and manifolds using mixture model cnns](https://arxiv.org/abs/1611.08402) (CVPR 2017)
1. etc: by "A Comprehensive Survey on Graph Neural Networks" P5;P7的表格分别列举了一些spatial-based GCN 

- **Together with sampling strategies, the computation can be performed in a batch of nodes instead of the whole graph** (GraphSAGE and LGCN)


## 谱上与空间GCN的比较：Comparison Between Spectral and Spatial Models
- **by "A Comprehensive Survey on Graph Neural Networks"**
- **bridges:** The graph convolution defined by 1stChebNet(semi-supervised GCN) is localized in space. It bridges the gap between spectral-based methods and spatial-based methods. --P2
- **Drawbacks** to spectral based models. We illustrate this in the following from three aspects, efficiency, generality and flexibility. --P11 

1. **Efficiency**</br>
**基于谱的模型** 或者需要计算特征向量，或者需要同时处理整个graph，这样的情况下，模型的计算量将随着graph size 显著的增加</br>
**基于空间的模型** 通过聚合临近节点的特征，直接在graph domain进行卷积计算，因此具有处理large graph的潜力。另外，可以以批次处理节点，而不是整个graph。再另外，随着临近节点的增加，可以使用采样策略来提高效率----参见后文 [改善GCN在训练方面的缺陷: Training Methods](#改善gcn在训练方面的缺陷-training-methods)</br>
1. **Generality**</br>
**基于谱的模型** 假设在固定的graph上进行训练，很难泛化到其他的新的或者不同的graph上</br>
**基于空间的模型** 以node为单位, 执行graph convolution计算，因此训练得到的权重(weights)可以轻易的共享到其他的node或者graph</br>
1. **Flexibility**</br>
**基于谱的模型** 受限于无向图，但是却没有在有向图上的关于拉普拉斯矩阵(Laplacian matrix)清晰的定义。因此，若将基于谱的方法应用在有向图上，需要先将有向图转化为无向图</br>
**基于空间的模型** 处理多源输入更加灵活，这里的多源输入可以指：edge features or edge directions, etc</br>
**关于edge features**, 参见下文 [输入含有边特征的GNN：input allow edge features](#输入含有边特征的gnninput-allow-edge-features)</br>


## 改善GCN在训练方面的缺陷: Training Methods
- **by "A Comprehensive Survey on Graph Neural Networks"**
- Comparison Between Spectral and Spatial Models P11
- [1stChebNet(semi-supervised GCN)](https://arxiv.org/abs/1609.02907)：the main drawback of 1stChebNet is that the computation cost increases exponentially with the increase of the number of 1stChebNet layers during batch training. Each node in the last layer has to expand its neighborhood recursively across previous layers.
1. [Fastgcn: fast learning with graph convolutional networks via importance sampling (ICLR 2018)](https://arxiv.org/abs/1801.10247)</br> 
assume the rescaled adjacent matrix A comes from a sampling distribution. 
1. [Stochastic training of graph convolutional networks with variance reduction (ICML 2018)](https://arxiv.org/abs/1710.10568)</br> 
reduce the receptive field size of the graph convolution to an arbitrary small scale by sampling neighborhoods and using historical hidden representations.
1. [Adaptive sampling towards fast graph representation learning (NIPS 2018)](https://arxiv.org/abs/1809.05343) </br>
propose an adaptive layer-wise sampling approach to accelerate the training of 1stChebNet, where sampling for the lower layer is conditioned on the top one. 

- **by "Graph Neural Networks: A Review of Methods and Applications"**
- Training Methods P9
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

- **by "Deep Learning on Graphs: A Survey"**
- Accelerating by Sampling P8
1. [Inductive representation learning on large graphs (NIPS 2017)](https://arxiv.org/abs/1706.02216)</br>
1. Graph convolutional neural networks for web-scale recommender systems
1. [Fastgcn: fast learning with graph convolutional networks via importance sampling (ICLR 2018)](https://arxiv.org/abs/1801.10247)</br>
1. [Stochastic training of graph convolutional networks with variance reduction (ICML 2018)](https://arxiv.org/abs/1710.10568)</br>


## Graph Attention Networks
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. [Graph Attention Network (GAT)](https://arxiv.org/abs/1710.10903)(ICLR 2017) [[tf code]](https://github.com/PetarV-/GAT)
1. [Gaan:Gated attention networks for learning on large and spatiotemporal graphs](https://arxiv.org/abs/1803.07294)
1. [Graph classification using structural attention](http://ryanrossi.com/pubs/KDD18-graph-attention-model.pdf)(ACM SIGKDD 2018)
1. [Watch your step: Learning node embeddings via graph attention](https://arxiv.org/abs/1710.09599)(NIPS 2018)


## Gated graph neural network 
- **by "Graph Neural Networks: A Review of Methods and Applications"**
1. Gated graph sequence neural networks (arXiv 2016)
1. Improved semantic representations from tree-structured long short-term memory networks (IJCNLP 2015)
1. Conversation modeling on reddit using a graph-structured lstm (TACL 2018)
1. Sentence-state lstm for text representation (ACL 2018)
1. Semantic object parsing with graph lstm (ECCV 2016)



## Residual and Jumping Connections/Skip Connections
- by yaya:考虑到CNN中residual network增加网络层数, 使得性能的提升, 这里尝试使用residual 也是为了在增加网络层数的基础上,使得性能更好. 参见下文：[Go deeper?]()
- **by "Deep Learning on Graphs: A Survey"** -- P7 Residual and Jumping Connections
1. Semi-supervised classification with graph convolutional networks (ICLR 2017)
1. Column networks for collective classification (AAAI 2017)
1. Representation learning on graphs with jumping knowledge networks (ICML 2018)
- **by "Graph Neural Networks: A Review of Methods and Applications"** -- P9 Skip Connections
1. Semi-supervised user geolocation via graph convolutional networks (ACL 2018)
1. Representation learning on graphs with jumping knowledge networks (ICML 2018)



## Graph Auto-encoders
<div align=center><img width="400" height="200" src="https://github.com/ShiYaya/graph/blob/master/images/graph-auto-encoder.png"/></div>

- **by "A Comprehensive Survey on Graph Neural Networks"**</br>
- network embedding算法可以分类为:1.matrix factorization 2.random walks 3. deep learning. Graph Auto-encoders是deep learning的一类方法. -- P2</br>
-  Network embedding是为了将node embedding 转化到低维的向量空间，通过保存网络的拓扑结构与节点内容信息，接下来的graph分析任务(比如，分类，聚类和推荐等)可以被应用于现有的机器学习任务(如SVM for classification)</br>

1. [Variational graph auto-encoders (GAE)](https://arxiv.org/abs/1611.07308) [[tkipf/code]](https://github.com/tkipf/gae) [[tf code]](https://github.com/limaosen0/Variational-Graph-Auto-Encoders)</br>
 used in link prediction task in citation networks</br>
 encoder对node embedding进行更新,decoder对A(adjacency matrix)进行更新
1. [Adversarially regularized graph autoencoder for graph embedding (ARGA)](https://arxiv.org/abs/1611.07308) [[tf code]](https://github.com/Ruiqi-Hu/ARGA)
1. [Learning deep network representations with adversarially regularized autoencoders (NetRA)](http://www.cs.ucsb.edu/~bzong/doc/kdd-18.pdf)
1. [Deep neural networks for learning graph representations (DNGR)](https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf) [[matlab code]](https://github.com/ShelsonCao/DNGR)
1. [Structural deep network embedding (SDNE)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf) [[python code]](https://github.com/suanrong/SDNE)
1. [Deep recursive network embedding with regular equivalence (DRNE)](http://pengcui.thumedialab.com/papers/NE-RegularEquivalence.pdf)(https://github.com/tadpole/DRNE)



## Graph Generative Networks
- **by "A Comprehensive Survey on Graph Neural Networks"**</br>
- factor the generation process as forming nodes and edges alternatively
1. [Graphrnn: A deep generative model for graphs](https://arxiv.org/abs/1802.08773) (ICML2018) [[tf code]](https://github.com/snap-stanford/GraphRNN)
1. [Learning deep generative models of graphs](https://arxiv.org/abs/1803.03324) (ICML2018)
- employ generative adversarial training
1. [Molgan: An implicit generative model for small molecular graphs](https://arxiv.org/pdf/1805.11973.pdf) (arXiv:1805.11973 2018)
1. [Net-gan: Generating graphs via random walks](https://arxiv.org/abs/1803.00816) (ICML2018)


## GCN Based Graph Spatial-Temporal Networks
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. [Diffusion convolutional recurrent neural network: Data-driven traffic forecasting (DCRNN)](https://arxiv.org/abs/1707.01926) (ICLR 2018)
1. [Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting (CNN-GNN)](https://arxiv.org/abs/1709.04875) (IJCAI 2017) [[tf code](https://github.com/VeritasYin/STGCN_IJCAI-18)
1. [Spatial temporal graph convolutional networks for skeleton-based action recognition (ST-GCN)](https://arxiv.org/abs/1801.07455) (AAAI 2018) [[pytorch code]](https://github.com/yysijie/st-gcn)
1. [Structural-rnn:Deep learning on spatio-temporal graphs (Structural-RNN)](https://arxiv.org/abs/1511.05298)  (CVPR 2016) [[theano code]](https://github.com/asheshjain399/RNNexp)
- **by yaya**
- 这两篇都是skeleton-based action recognition
1. [Skeleton-Based Action Recognition with Spatial Reasoning and Temporal Stack Learning](https://arxiv.org/abs/1805.02335) (ECCV 2018)



## 输入含有边特征的GNN：Input allow edge features 
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
- **Overview**
- **by "Graph Neural Networks: A Review of Methods and Applications"**
<div align=center><img src="https://github.com/ShiYaya/graph/blob/master/images/gcn-in-test-application.png"/></div>

1. [Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/abs/1703.04826)</br>
* [[官方code(theano 0.8.2,lasagne 0.1)]](https://github.com/diegma/neural-dep-srl)  [[复现pytorch]](https://github.com/kervyRivas/Graph-convolutional)
* [专知讲解](https://mp.weixin.qq.com/s/c6ZhSk4r3pvnjHsvpwkkSw)
* by yaya:阅读该篇文章主要是来源于这篇将图卷积用于图像描述的文章:Exploring Visual Relationship for Image Captioning</br>
这两篇文章采用的图卷积公式都是一样的，但是我认为很奇怪，而且b是如何由edge获得的，将进一步阅读代码，稍后解释。</br>
<div align=center><img src="https://github.com/ShiYaya/graph/blob/master/images/gcn%2Bformulation.png"/></div>

1. [Graph Convolutional Encoders for Syntax-aware Neural Machine Translation](https://arxiv.org/pdf/1704.04675)

## 计算机视觉
- **Overview**
- **by "Graph Neural Networks: A Review of Methods and Applications"**
<div align=center><img src="https://github.com/ShiYaya/graph/blob/master/images/gcn-in-image-application.png"/></div>

### Scene graph generation
- **by "A Comprehensive Survey on Graph Neural Networks"**
- **detect and recognize objects and predict semantic relationships between pairs of objects**
1. Scene graph generation by iterative message passing (CVPR 2017)
1. Graph r-cnn for scene graph generation (ECCV 2018)
1. Factorizable net: an efficient subgraph-based framework for scene graph generation (ECCV 2018)

- **generating realistic images given scene graphs**
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. Image generation from scene graphs (arXiv preprint, 2018)

### Point clouds classification and segmentation
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. Dynamic graph cnn for learning on point clouds(arXiv preprint 2018)
1. Large-scale point cloud semantic segmentation with superpoint graphs (CVPR 2018)
1. Rgcnn: Regularized graph cnn for point cloud segmentation (arXiv preprint 2018)


### Action recognition
- **by "A Comprehensive Survey on Graph Neural Networks"**
- detects the locations of human joints in video clips
1. [Spatial temporal graph convolutional networks for skeleton-based action recognition (ST-GCN)](https://arxiv.org/abs/1801.07455) (AAAI 2018) [[pytorch code]](https://github.com/yysijie/st-gcn)
1. [Structural-rnn:Deep learning on spatio-temporal graphs (Structural-RNN)](https://arxiv.org/abs/1511.05298)  (CVPR 2016) [[theano code]](https://github.com/asheshjain399/RNNexp)
- **by yaya**
1. [Skeleton-Based Action Recognition with Spatial Reasoning and Temporal Stack Learning](https://arxiv.org/abs/1805.02335) (ECCV 2018)


### Image classification
- **by "Graph Neural Networks: A Review of Methods and Applications"**
1. [Few-shot learning with graph neural networks](https://arxiv.org/abs/1711.04043) (ICLR 2018) [[code]](https://github.com/vgsatorras/few-shot-gnn)
1. [Zero-shot recognition via semantic embeddings and knowledge graphs](https://arxiv.org/abs/1803.08035) (CVPR 2018)
1. [Multi-label zero-shot learning with structured knowledge graphs](https://arxiv.org/abs/1711.06526) (arXiv preprint 2017)
1. [Rethinking knowledge graph propagation for zero-shot learning](https://arxiv.org/abs/1805.11724)(arXiv preprint 2018)
1. [The more you know: Using knowledge graphs for image classification](https://arxiv.org/abs/1612.04844) (arXiv preprint 2016)
- **by yaya**
1. [Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning](https://arxiv.org/abs/1805.10002) [[tf code]](https://github.com/csyanbin/TPN)


### Few-shot 
- **by "A Comprehensive Survey on Graph Neural Networks"**
- image classification
1. [Few-shot learning with graph neural networks](https://arxiv.org/abs/1711.04043) (ICLR 2018) [[code]](https://github.com/vgsatorras/few-shot-gnn)
- 3d action recognition
1. [Neural graph matching networks for fewshot 3d action recognition](http://openaccess.thecvf.com/content_ECCV_2018/html/Michelle_Guo_Neural_Graph_Matching_ECCV_2018_paper.html) (ECCV 2018)
- **by yaya**
- image classification
1. [Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning](https://arxiv.org/abs/1805.10002) [[tf code]](https://github.com/csyanbin/TPN)

### Zero-shot
1. [Zero-shot recognition via semantic embeddings and knowledge graphs](https://arxiv.org/abs/1803.08035) (CVPR 2018)
1. [Multi-label zero-shot learning with structured knowledge graphs](https://arxiv.org/abs/1711.06526) (arXiv preprint 2017)
1. [Rethinking knowledge graph propagation for zero-shot learning](https://arxiv.org/abs/1805.11724)(arXiv preprint 2018)

### Semantic segmentation
- **by "A Comprehensive Survey on Graph Neural Networks"**
1. 3d graph neural networks for rgbd semantic segmentation (CVPR 2017)
1. Syncspeccnn: Synchronized spectral cnn for 3d shape segmentation (CVPR 2017)
- **by "Graph Neural Networks: A Review of Methods and Applications"**
1. Semantic object parsing with graph lstm (ECCV 2016)
1. Interpretable structure-evolving lstm (CVPR 2017)
1. Large-scale point cloud semantic segmentation with superpoint graphs(arXiv preprint 2017)
1. Dynamic graph cnn for learning on point clouds(arXiv preprint 2018)
1. 3d graph neural networks for rgbd semantic segmentation (CVPR 2017)

### Visual question answer
- **by "Graph Neural Networks: A Review of Methods and Applications"**
1. **A simple neural network module for relational reasoning.**
*Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap.* NIPS 2017. [paper](https://arxiv.org/pdf/1706.01427.pdf)
1. **Graph-Structured Representations for Visual Question Answering.**
*Damien Teney, Lingqiao Liu, Anton van den Hengel.* CVPR 2017. [paper](https://arxiv.org/pdf/1609.05600.pdf)
1. **Out of the Box: Reasoning with Graph Convolution Nets for Factual Visual Question Answering.**
*Medhini Narasimhan, Svetlana Lazebnik, Alexander Schwing.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7531-out-of-the-box-reasoning-with-graph-convolution-nets-for-factual-visual-question-answering.pdf)
1. **Learning Conditioned Graph Structures for Interpretable Visual Question Answering.**
*Will Norcliffe-Brown, Efstathios Vafeias, Sarah Parisot.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1806.07243)  [[code]](https://github.com/aimbrain/vqa-project)
1. **Deep reasoning with knowledge graph for social relationship understanding.**

### Object detection
- **by "Graph Neural Networks: A Review of Methods and Applications"**
1. Relation networks for object detection (CVPR 2018)
1. Learning region features for object detection (arXiv preprint 2018)

### Interaction detection
- **by "Graph Neural Networks: A Review of Methods and Applications"**
1. [Learning humanobject interactions by graph parsing neural networks] (arXiv preprint 2018)
1. Structural-rnn:Deep learning on spatio-temporal graphs (CVPR 2016)

### Region classification
- **by "Graph Neural Networks: A Review of Methods and Applications"**
1. Iterative visual reasoning beyond convolutions (arXiv preprint 2018)

### Social Relationship Understanding
1. Deep reasoning with knowledge graph for social relationship understanding

## **Library**

3. https://github.com/rusty1s/pytorch geometric
4. https://www.dgl.ai/

- **geometric learning library [[github]](https://github.com/rusty1s/pytorch_geometric)** 
in PyTorch named PyTorch Geometric, which implements serveral graph neural networks including ChebNet, 1stChebNet, GraphSage, MPNNs, GAT and SplineCNN.
- **Deep Graph Library (DGL) [[website]](https://www.dgl.ai/) [[github]](https://github.com/dmlc/dgl)**
provides a fast implementation of many graph neural networks with a set of functions on top of popular deep learning platforms such as PyTorch and MXNet
- **graph_nets [[github]](https://github.com/deepmind/graph_nets)**


## Other application

- **by "Graph Neural Networks: A Review of Methods and Applications"**
<div align=center><img src="https://github.com/ShiYaya/graph/blob/master/images/gcn-in-other-application.png"/></div>



## Open problems and future direction
**一. Go deeper?**
- **by "Graph Neural Networks: A Review of Methods and Applications"  &  "A Comprehensive Survey on Graph Neural Networks"**</br>
(1)当前的gnn的层数大都很浅,这是因为，随着网络层数的增加，representation of nodes将趋于平滑,换句话说，图卷积本质上是使相邻节点的表达更加接近，从而在理论上来说，在无限次卷积的情况下，所有节点的表达都将会收敛于一个稳定的点，节点特征的可区分性与信息的丰富性将会损失。在图结构数据上的网络增加层数是否是一个好的策略仍然是一个开放性的问题。[[Deeper insights into graph convolutional networks for semi-supervised learning]](https://arxiv.org/abs/1801.07606)</br>
(2)when tack k layers, each node will aggregate more information from neighborhoods k hops away. 若临近节点有噪声，将会随着层数的增加，噪声信息也会指数级增加. P9 by "Graph Neural Networks: A Review of Methods and Applications"--skip connection</br>
受到传统deep neural networks在增加网络深度上取得的显著结果，一些研究者也尝试解决GNN中的网络层数难以加深的问题：</br>

- **by "A Comprehensive Survey on Graph Neural Networks"**</br>
1. Gated graph sequence neural networks (arXiv 2016)</br>
1. Deeper insights into graph convolutional networks for semi-supervised learning (arXiv preprint 2018)</br>
- **by "Graph Neural Networks: A Review of Methods and Applications"**</br>
1. Semi-supervised user geolocation via graph convolutional networks (ACL 2018)
1. Representation learning on graphs with jumping knowledge networks (ICML 2018)

***二. Non-structural Scenarios->generate graph from raw data***
- **by "Graph Neural Networks: A Review of Methods and Applications"**</br>
- 虽然上文讨论了graph在非结构化场景(image, text)中的应用, 但是目前却没有从原始数据中来生成graph的最优的方法. in image domain, 一些工作利用CNN来获取特征映射, 然后对其进行采样得到的超像素作为节点, 其他的也有提取object作为节点. in test domain, 一些工作利用syntactic trees作为syntactic graphs, 另外其他的工作直接采用全连接graphs. </br>
因此找到最佳的graph generation approach将提供更广泛的领域, GNN可以在这些领域中做出贡献。


**三. Dynamic graphs**
- **by "Deep Learning on Graphs: A Survey"**
在社交网络中，存在新的人加入，或者已存在的人退出社交网络，这样的graph是动态的，而当前提出的方法都是建立在 static graph.</br> 
How to model the evolving characteristics of dynamic graphs and support incrementally updating model parameters largely remains open in the literature.</br>
- Some preliminary works try to tackle this problem using Graph RNN architectures with encouraging results
1. Dynamic graph neural networks (arXiv preprint 2018)
1. Dynamic graph convolutional networks (arXiv preprint 2017)


**四. Different types of graphs**
- **by "Deep Learning on Graphs: A Survey"**
- **homogeneous graphs**
Heterogeneous network embedding via deep architectures
- **Signed networks**
Signed graph convolutional network
- **Hyper graphs**
Structural deep embedding for hyper-networks (AAAI 2018)

**五. Interpretability**
- **by "Deep Learning on Graphs: A Survey"**
- 由于graph经常与其他学科相关联，因此解释图形的深度学习模型对于决策问题至关重要，例如，在药物或者疾病相关的问题，可解释性对于将计算机实验转化为临床应用至关重要。然而，由于图中的节点和边缘是紧密相互关联的， 因此基于图形的深度学习的可解释性甚至比其他黑匣子模型更具挑战性

**六. Compositionality**
- **by "Deep Learning on Graphs: A Survey"**
- 很多存在的方法可以组合到一起，例如将GCN作为GAEs或者Graph RNNs里的layer, 除了设计新的building blocks, 如何将现有的结构以某种原则组合到一起也是一个很有趣的方向, 最近的工作, [Graph Networks](https://arxiv.org/abs/1806.01261)进行了尝试, 重点介绍了GNNS和GCNS通用框架在关系推理问题中的应用。


**七. Scalability->Can gnn handle large graphs?**
- **by "Graph Neural Networks: A Review of Methods and Applications"**</br>
Scaling up GNN is difficult because many of the core steps are computational consuming in big data environment:1. graph不是规则的欧式空间，receptive filed(neighborhood structure) 对于每个node也是不同的，因此很难对节点进行批次训练. 2. 当处理 large graph时，计算graph Laplacian也很困难.
- **by yaya**
我觉得这样的说法是不对的，由上文的分析中可以看出，只是谱方法需要计算graph Laplacian，

- **by "A Comprehensive Survey on Graph Neural Networks"**</br>
当gcn的堆叠多层时，一个节点的最终状态将由很多临近节点((1~k)-hop neighbors)的状态所决定, 在反向传播时的计算量将会很大。当前为了提高模型的效率提出了两类方法fast sampling and sub-graph training, but still not scalable enough to handle deep architectures with large graphs</br>
**fast sampling**</br>
1. [Fastgcn: fast learning with graph convolutional networks via importance sampling (ICLR 2018)](https://arxiv.org/abs/1801.10247)</br> 
1. [Stochastic training of graph convolutional networks with variance reduction (ICML 2018)](https://arxiv.org/abs/1710.10568)</br> 
**sub-graph training**</br>
1. [Inductive representation learning on large graphs (NIPS 2017)](https://arxiv.org/abs/1706.02216)</br>
1. [Large-scale learnable graph convolutional networks](https://arxiv.org/abs/1808.03965) (ACM 2018)

- **by yaya：**我觉得这样说，是从deep gnn的角度来说，这样就没有讲清shallow gnn是否可以应用于large graph

- ***yaya conclution:*** 基于公式X'=AXW的GCN网络，需要将entire graph输入网络中进行计算，不能以节点为单位进行batch运算，计算量大，对于设计了sub-graph的网络，局限性可能在于邻近节点很多，若网络也很深，计算量将也会很大

**八. Receptive Field**
- **by "A Comprehensive Survey on Graph Neural Networks"**</br>
- 这里的Receptive Field是参考了论文"Deep Learning on Graphs: A Survey"中的Accelerating by Sampling这一节, 目的也是在于加速训练
- 一个node的可接受域是指它本身以及its neighbors, But the number of neighbors is very different, from one to thousands. 遵循power law
distribution. 因此采样策略被提出来, 如何选择节点的有代表性的接收域仍有待探索
1. Inductive representation learning on large graphs (NIPS 2017)
1. Learning convolutional neural networks for graphs (ICML 2016)
1. Large-scale learnable graph convolutional networks (ACM　SIGKDD 2018)




