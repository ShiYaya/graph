# graph
####关于graph可做的方向
- 图的任务也有很大的不同，可以是node-focused问题，如节点分类和链接预测，也可以是graph-focused问题，如图分类和图生成。不同的结构和任务需要不同的模型架构来处理特定的问题。   
据此，现在要做的方向是视频描述，后边可以做一些图像生成的，graph与GAN结合的一些工作，但是似乎是有了。。再具体看看。


### 近期Graph的学习任务 
- semantic role labeling 代码
- Hierarchical Graph Representation Learning with Differentiable Pooling 代码
- Learning Conditioned Graph Structures for Interpretable Visual Question Answering 代码：github.com/aimbrain/vqa-project.
condition 是基于question.那么在视频描述中也可以基于监督学习本身带有的标签，在Inference时，则，直接利用训练好的graph参数（此处可以参考一下few-shot）
- 其他关于视觉问答的论文
- Graph Neural Networks: A Review of Methods and Applications提到了一些关于视觉问答的论文可以找出来，看一看，还有关于源代码的部分

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

## 谱上的图卷积发展
- [The Emerging Field of Signal Processing on Graphs](https://arxiv.org/pdf/1211.0053.pdf)
- [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)
- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), [[PyTorch Code]](https://github.com/xbresson/graph_convnets_pytorch/blob/master/README.md) [[TF Code]](https://github.com/mdeff/cnn_graph)
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), [[Code]](https://github.com/tkipf/gcn), [[Blog]](http://tkipf.github.io/graph-convolutional-networks/)


## GCN的应用

### 用图卷积网络( GCN)来做语义角色标注</br>
[Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/abs/1703.04826)</br>
* [[官方code(theano 0.8.2,lasagne 0.1)]](https://github.com/diegma/neural-dep-srl)  [[复现pytorch]](https://github.com/kervyRivas/Graph-convolutional)
* [专知讲解](https://mp.weixin.qq.com/s/c6ZhSk4r3pvnjHsvpwkkSw)
* by yaya:阅读该篇文章主要是来源于这篇将图卷积用于图像描述的文章:Exploring Visual Relationship for Image Captioning</br>
这两篇文章采用的图卷积公式都是一样的，但是我认为很奇怪，而且b是如何由edge获得的，将进一步阅读代码，稍后解释。</br>
<img src="https://github.com/ShiYaya/graph/blob/master/images/gcn%2Bformulation.png" width="200" height="100" ></br>
