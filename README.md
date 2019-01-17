# graph
### 关于graph可做的方向
- 图的任务也有很大的不同，可以是node-focused问题，如节点分类和链接预测，也可以是graph-focused问题，如图分类和图生成。不同的结构和任务需要不同的模型架构来处理特定的问题。   
据此，现在要做的方向是视频描述，后边可以做一些图像生成的，graph与GAN结合的一些工作，但是似乎是有了。。再具体看看。


### 近期Graph的学习任务 
- semantic role labeling 代码
- Hierarchical Graph Representation Learning with Differentiable Pooling 代码
- Learning Conditioned Graph Structures for Interpretable Visual Question Answering 代码：github.com/aimbrain/vqa-project.
condition 是基于question.那么在视频描述中也可以基于监督学习本身带有的标签，在Inference时，则，直接利用训练好的graph参数（此处可以参考一下few-shot）
- 其他关于视觉问答的论文
- Graph Neural Networks: A Review of Methods and Applications提到了一些关于视觉问答的论文可以找出来，看一看，还有关于源代码的部分

### 待学习：
- 带边信息的图(Edge-informative Graph)
每条边都有信息，比如权值或边的类型。例如G2S和R-GCN。
1. **Graph-to-Sequence Learning using Gated Graph Neural Networks.**
*Daniel Beck, Gholamreza Haffari, Trevor Cohn.* ACL 2018. [paper](https://arxiv.org/pdf/1806.09835.pdf)
1. **Modeling Relational Data with Graph Convolutional Networks.**
*Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling.* ESWC 2018. [paper](https://arxiv.org/pdf/1703.06103.pdf)

- 使用不同训练方法的图变体
1. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling.**
*Jie Chen, Tengfei Ma, Cao Xiao.* ICLR 2018. [paper](https://arxiv.org/pdf/1801.10247.pdf)



### 问题
在Graph Neural Networks: A Review of Methods and Applications论文P16中，也提到了语义角色标注这篇文章，说"special variant of the GCN "；
在语义角色标注的本文中，也提到了More formally....这一处需要再理解一些。




### 通过池化，提出了一种graph classification的方法</br>
[Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf),[code](https://github.com/RexYing/diffpool)


## 总结性质的
### github上的某篇总结-介绍了相关的论文、博客、以及研究者</br>
https://github.com/sungyongs/graph-based-nn</br>
https://github.com/thunlp/GNNPapers</br>

### 综述论文
- [Deep Learning on Graphs: A Survey](https://arxiv.org/abs/1812.04202)  
[[新智元解读]](https://mp.weixin.qq.com/s/eelcT5x_kWC0dDt0_Ph4qg)
- [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)  
[[新智元解读]](https://mp.weixin.qq.com/s/h4jQWJlQV2Ew3SpuF8k5Hw)
- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)
- [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261.pdf)

## 谱上的图卷积发展:spectral-based graph convolutional networks

- 以下四篇是按照时间轴，依次在前一篇的文章上进行改进的
1. [The Emerging Field of Signal Processing on Graphs](https://arxiv.org/pdf/1211.0053.pdf)
1. [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)
1. [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), [[PyTorch Code]](https://github.com/xbresson/graph_convnets_pytorch/blob/master/README.md) [[TF Code]](https://github.com/mdeff/cnn_graph)
1. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), [[Code]](https://github.com/tkipf/gcn), [[Blog]](http://tkipf.github.io/graph-convolutional-networks/)

- 以下三篇是在<A Comprehensive Survey on Graph Neural Networks>这篇综述中提到的另外三篇
1. [Deep convolutional networks on graph-structured data](https://arxiv.org/abs/1506.05163)
1. [Adaptive graph convolutional neural networks](https://arxiv.org/abs/1801.03226)  可接受任意图结构和规模的图作为输入
1. [Cayleynets: Graph convolutional neural networks with complex rational spectral filters](https://arxiv.org/abs/1705.07664)

- 谱上的图卷积网络的缺陷：   
**spectral methods usually handle the whole graph simultaneously and are difficult to parallel or scale to large graphs**

## 空间上的图卷积：spatial-based graph convolutional networks

1. GraphSAGE[Inductive representation learning on large graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
1. [Geometric deep learning on graphs and manifolds using mixture model cnns](http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf)
1. [Learning convolutional neural networks for graphs](http://proceedings.mlr.press/v48/niepert16.pdf)
1. LGCN[Large-scale learnable graph convolutional networks](https://dl.acm.org/citation.cfm?id=3219947)

- GraphSAGE and LGCN : **Together with sampling strategies, the computation can be performed in a batch of nodes instead of the whole graph

## GCN的应用

### 用图卷积网络( GCN)来做语义角色标注</br>
[Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/abs/1703.04826)</br>
* [[官方code(theano 0.8.2,lasagne 0.1)]](https://github.com/diegma/neural-dep-srl)  [[复现pytorch]](https://github.com/kervyRivas/Graph-convolutional)
* [专知讲解](https://mp.weixin.qq.com/s/c6ZhSk4r3pvnjHsvpwkkSw)
* by yaya:阅读该篇文章主要是来源于这篇将图卷积用于图像描述的文章:Exploring Visual Relationship for Image Captioning</br>
这两篇文章采用的图卷积公式都是一样的，但是我认为很奇怪，而且b是如何由edge获得的，将进一步阅读代码，稍后解释。</br>
<img src="https://github.com/ShiYaya/graph/blob/master/images/gcn%2Bformulation.png" width="200" height="100" ></br>
