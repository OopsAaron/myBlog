---
title: 2020-11-24-transformer优化-temp
mathjax: true
date: 2020-11-24 19:48:40
tags:
top:
categories:
description: transformer模型改进优化总结
---





想法：



总体来说，基于原始的transformer模型，在这个基础上综合增加其它论文的创新点，以及最好能够有自己的创新点



思路有两个：

1. 通过现有的模型，去解决没有解决过的任务。比如NLP的模型去解决CV的任务
2. 在现有的模型中去添加自己的想法，使得有更好的性能 ，如transformer-XL对transformer的改进





一些想法：

1. 一些在bert上进行优化的模型是否可以运用到修改transformer的attention结构中。例如ConvBERT,AlBERT

   ALBERT相比原始BERT其实更适合浅层Transformer，也可以作为之后的尝试方向。

2. 降低transformer模型的计算量，已经存在的模型，模型剪枝、量化、蒸馏等精简技术，又或者修改 Attention 结构

3. 解决自回归的问题，之前读过teacher-forcing 模型来进行改进

4. 之前改进的模型如果用在图像领域，可以用在文本领域，如果这样做的话，是否有创新性呢？？

5. 感觉最近的论文对于transformer的稀疏性研究挺多  ，是否可以借鉴几篇 。 对transformer进行优化

6. 将transformer中的部分结构换成LSTM是否会效果好一些呢 ？ 结合attention和LSTM

7. 重点基于transformer-XL和Reformer这些关注点高的论文

8. 从经典论文引用的论文中去找到新方法去结合自己的模型

9. BERT以及其它的模型，可以不用这么多层数，不用这么多的数据集，去设计不同的自监督任务去改进模型。

   可以像用word2vec一样使用这些pretrain模型，然后去专注task相关的研究（在你能接受的数据和计算资源范围内。这样是否可以呢 ？？ 

   比如只有6层的BERT-base中，MLM策略是采用的自己设计的，或者融合其它模型的MLM策略，与BERT-base进行比较，发现效果有所提升，是否就可以发表论文？？

   因为设计的自监督任务必须要与下游任务尽量相关的，这样才能在预训练阶段学到更多东西

10. 对于一些优化器或者激活函数或者norm函数进行优化，（使用最新模型使用的优化器）可能会对模型效果有所提升

11. 有空读读word2vec的内容，从之前的模型中去结合形成创新点

12. 网上是否有综合了transformer-base模型的github项目。类似于hugging face对于BERT-base模型

13. 将decoder的自回归变成非自回归形式的。参考论文<Deep Encoder, Shallow Decoder: Reevaluating the Speed-Quality Tradeoff in Machine Translation> 2020年论文，无代码

14. exposure bias问题的解决，参考<Scheduled Sampling for Transformers>，8.1分享。有torch代码（非官方）

15. 将一些部分设置为adaptive 动态的。

    



总体就是以一个模型（如transformer-XL）为基础，在上面增加其它的思想，增加创新点





计划：

transformer-XL可以增强transformer处理长文本的能力，但是同时它的复杂度依然是n2，所以以transformer-XL为基础，来优化结构，降低复杂度，是可以解决的一个思路，同时如果能够解决exposure bias问题更好





 以transformer-XL为基础进行改进，有pytorch代码。解决机器翻译问题，或者看邱锡鹏教授实验室的研究方向

[https://github.com/kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl)

对transformer-XL进行压缩，减少memory。   或者参考XLNET， 也是结合transformer-xl

结合BERT的一些思想， 或者其它的压缩模型思想

线性核transformer进行结合 ， paperweekly 苏剑林的提议， 结合到transformer-xl中



将层数降低比较是否可以？







今年ICLR2020已经有一些工作对天transformer-XL进行改进 。借鉴查看



------





疑问： 

如果用bert去做图像的任务，会有以下问题

1. BERT 训练需要的资源多，可能会训练的周期长。解决方法：将参数量减少，层数减少

2. bert 主要是进行自监督，也就是处理的是没有标签的数据。这样的数据在nlp很多，但是在图像上很多是有标签的

   

   
