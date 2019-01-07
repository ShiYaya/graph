# graph

### 通过池化，提出了一种graph classification的方法</br>
[Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf)</br>
code:https://github.com/RexYing/diffpool</br>

### 谱上的图卷积发展
- [The Emerging Field of Signal Processing on Graphs](https://arxiv.org/pdf/1211.0053.pdf)
- [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)
- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), [[PyTorch Code]](https://github.com/xbresson/graph_convnets_pytorch/blob/master/README.md) [[TF Code]](https://github.com/mdeff/cnn_graph)
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), [[Code]](https://github.com/tkipf/gcn), [[Blog]](http://tkipf.github.io/graph-convolutional-networks/)




### github上的某篇总结-介绍了相关的论文、博客、以及研究者</br>
https://github.com/sungyongs/graph-based-nn</br>

### 用图卷积网络( GCN)来做语义角色标注</br>
[Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling](https://arxiv.org/abs/1703.04826)</br>
* 官方code(theano 0.8.2,lasagne 0.1):https://github.com/diegma/neural-dep-srl</br>
* 复现pytorch:https://github.com/kervyRivas/Graph-convolutional</br>
* 专知讲解：https://mp.weixin.qq.com/s/c6ZhSk4r3pvnjHsvpwkkSw</br>
* by yaya:阅读该篇文章主要是来源于这篇将图卷积用于图像描述的文章:Exploring Visual Relationship for Image Captioning</br>
这两篇文章采用的图卷积公式都是一样的，但是我认为很奇怪，而且b是如何由edge获得的，将进一步阅读代码，稍后解释。</br>
<img src="https://github.com/ShiYaya/graph/blob/master/images/gcn%2Bformulation.png" width="200" height="100" ></br>
