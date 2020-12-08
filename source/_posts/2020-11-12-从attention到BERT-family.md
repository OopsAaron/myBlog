---
title: 2020-11-12-从attention到BERT-family
mathjax: true
date: 2020-11-12 12:54:33
tags:
top:
categories:
description: 对最近一段时间读的论文以及自己的对NLP领域transformer系列模型的了解做一个总结
---

### 前言

做NLP领域的应该没人不知道BERT，之前开组会的时候讲到过BERT的一篇改进论文，导师建议对transformer、BERT以及预训练模型发展做个总结综述，作为下次汇报的内容。于是这几天在网上参考了一些博客，还有大佬邱锡鹏老师和刘群老师关于预训练模型的报告，以及结合自己读的一些相关论文，对整个模型的发展演变梳理了一下，汇总如下

### attention以及self-attention



#### 提出

借鉴了人类的注意力机制



![人类的视觉注意力](E:\myBlog\source\_posts\516P2JBYrgOToyU.jpg)





人类视觉通过快速扫描全局图像，获得注意力焦点，而后对这一区域投入更多注意力资源，以获取更多所需要关注目标的细节信息，而抑制其他无用信息。这是人类利用有限的注意力资源从大量信息中快速筛选出高价值信息的手段

深度学习中的注意力机制从本质上讲和人类的选择性视觉注意力机制类似，核心目标也是从众多信息中选择出对当前任务目标更关键的信息。



#### Encoder-Decoder框架

一般的seq2seq模型



![抽象的文本处理领域的Encoder-Decoder框架](E:\myBlog\source\_posts\ohBVdz1KDn4ZcOb.png)





对于句子对<Source,Target>，我们的目标是给定输入句子Source，期待通过Encoder-Decoder框架来生成目标句子Target。Source和Target分别由各自的单词序列构成：

![img](E:\myBlog\source\_posts\qrCwtas6iRVKYbA.png)



Encoder对输入句子Source进行编码，将输入句子通过非线性变换转化为中间语义表示C：



![img](E:\myBlog\source\_posts\M7EXx2KPgeHFQhL.png)



对于解码器Decoder来说，其任务是根据句子Source的中间语义表示C和之前已经生成的历史信息 y1,y2,y3...yi-1来生成i时刻要生成的单词yi：

![img](E:\myBlog\source\_posts\6uHkShXDNKOlBQ3.png)



每个yi都依次这么产生，那么看起来就是整个系统根据输入句子Source生成了目标句子Target。机器翻译、文本摘要、问答系统都是encoder-decoder框架。



#### 问题

![img](E:\myBlog\source\_posts\KNWS6Ax1uI285cH.png)

在生成目标句子的单词时，不论生成哪个单词，它们使用的输入句子Source的语义编码C都是一样的，没有任何区别。

而语义编码C是由句子Source的每个单词经过Encoder编码产生的，这意味着不论是生成哪个单词，y1,y2还是y3，其实句子Source中任意单词对生成某个目标单词yi来说影响力都是相同的，这是为何说这个模型没有体现出注意力的缘由。这类似于人类看到眼前的画面，但是眼中却没有注意焦点



没有引入注意力的模型在输入句子比较短的时候问题不大，但是如果输入句子比较长，此时所有语义完全通过一个中间语义向量来表示，单词自身的信息已经消失，可想而知会丢失很多细节信息，这也是为何要引入注意力模型的重要原因。



#### attention



目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。这意味着在生成每个单词yi的时候，原先都是相同的中间语义表示C会被替换成根据当前生成单词而不断变化的Ci（C是动态的，根据生成次词的不同，C也是不同的）。

**即由固定的中间语义表示C换成了根据当前输出单词来调整成加入注意力模型的变化的Ci**。



![引入注意力模型的Encoder-Decoder框架](E:\myBlog\source\_posts\oe6E7C9sUHrVuNZ.png)







即生成目标句子单词的过程成了下面的形式：



![img](E:\myBlog\source\_posts\q8ZFEu4BSI1lo9z.png)



而每个Ci可能对应着不同的源语句子单词的注意力分配概率分布





![img](E:\myBlog\source\_posts\TBg8KZhoyiOlUst.png)

实际中，Tom、chase、jerry都是被编码成512维的向量，所以权重相加之后Ctom应该也是一个向量



其中，f2函数代表Encoder对输入英文单词的某种变换函数，比如如果Encoder是用的RNN模型的话，这个f2函数的结果往往是某个时刻输入xi后隐层节点的状态值；g代表Encoder根据单词的中间表示合成整个句子中间语义表示的变换函数，一般的做法中，g函数就是对构成元素加权求和，即下列公式：



![img](E:\myBlog\source\_posts\W9SnjkNw3BVasdc.png)



其中，Lx代表输入句子Source的长度，aij代表在Target输出第i个单词时Source输入句子中第j个单词的注意力分配系数，而hj则是Source输入句子中第j个单词的语义编码。假设下标i就是上面例子所说的“
汤姆”
，那么Lx就是3，h1=f(“Tom”)，h2=f(“Chase”),h3=f(“Jerry”)分别是输入句子每个单词的语义编码，对应的注意力模型权值则分别是0.6,0.2,0.2，所以g函数本质上就是个加权求和函数。如果形象表示的话，翻译中文单词“汤姆”的时候，数学公式对应的中间语义表示Ci的形成过程类似图4。





![Attention的形成过程](E:\myBlog\source\_posts\LRJZMaUbN9VuxGr.png)





如何得到单词注意力分配概率Ｃ呢？

为了便于说明，我们假设对图2的非Attention模型的Encoder-Decoder框架进行细化，Encoder采用RNN模型，Decoder也采用RNN模型，这是比较常见的一种模型配置，则图2的框架转换为图5。



![RNN作为具体模型的Encoder-Decoder框架](E:\myBlog\source\_posts\ebTh6NzIdlVG1L2.png)





那么用下图可以较为便捷地说明注意力分配概率分布值的通用计算过程。



![注意力分配概率计算](E:\myBlog\source\_posts\TEXoxdzgrHOuiMQ.jpg)





:star:  利用的是i-1时刻的隐状态（作为query）去和h1，h2，h3求相似性

 对于采用RNN的Decoder来说，在时刻i，如果要生成yi单词，我们是可以知道Target在生成Yi之前的时刻i-1时，隐层节点i-1时刻的输出值Hi-1的，而我们的目的是要计算**生成Yi时**输入句子中的单词对Yi来说的注意力分配概率分布，那么**可以用Target输出句子i-1时刻的隐层节点状态Hi-1去一一和输入句子Source中每个单词对应的RNN隐层节点状态hj进行对比**，然后函数F的输出经过Softmax进行归一化就得到了符合概率分布取值区间的注意力分配概率分布数值。



#### Attention机制的本质思想





![Attention机制的本质思想](E:\myBlog\source\_posts\y3BToK96rsgUZfF.png)





我们可以这样来看待Attention机制：将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。

所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。即可以将其本质思想改写为如下公式：



![img](E:\myBlog\source\_posts\ipGlzuFcmS8n2VR.png)



其中，Lx=||Source||代表Source的长度，公式含义即如上所述。上文所举的机器翻译的例子里，因为在计算Attention的过程中，Source中的Key和Value合二为一，指向的是同一个东西，也即输入句子中每个单词对应的语义编码，所以可能不容易看出这种能够体现本质思想的结构。

当然，从概念上理解，把Attention仍然理解为从大量信息中有选择地筛选出少量重要信息并聚焦到这些重要信息上，忽略大多不重要的信息，这种思路仍然成立。聚焦的过程体现在权重系数的计算上，权重越大越聚焦于其对应的Value值上，即权重代表了信息的重要性，而Value是其对应的信息。





#### 计算过程

至于Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：第一个过程是根据Query和Key计算权重系数，第二个过程根据权重系数对Value进行加权求和。而第一个过程又可以细分为两个阶段：第一个阶段根据Query和Key计算两者的相似性或者相关性；第二个阶段对第一阶段的原始分值进行归一化处理；这样，可以将Attention的计算过程抽象为如图10展示的三个阶段。



![三阶段计算Attention过程](E:\myBlog\source\_posts\tzO9fhXUeBblmVG.png)





在第一个阶段，可以引入不同的函数和计算机制，根据Query和某个Key_i，计算两者的相似性或者相关性，最常见的方法包括：求两者的向量点积、求两者的向量Cosine相似性或者通过再引入额外的神经网络来求值，即如下方式：



![img](E:\myBlog\source\_posts\9xpPOa7ohFf3u1d.png)



第一阶段产生的分值根据具体产生的方法不同其数值取值范围也不一样，第二阶段引入类似SoftMax的计算方式对第一阶段的得分进行数值转换，一方面可以进行归一化，将原始计算分值整理成所有元素权重之和为1的概率分布；另一方面也可以通过SoftMax的内在机制更加突出重要元素的权重。即一般采用如下公式计算：

![img](E:\myBlog\source\_posts\XFW5tcSGjqBnIyN.png)



第二阶段的计算结果a_i即为value_i对应的权重系数，然后进行加权求和即可得到Attention数值：



![img](E:\myBlog\source\_posts\soa1M9LIPGi3krC.png)



通过如上三个阶段的计算，即可求出针对Query的Attention数值，目前绝大多数具体的注意力机制计算方法都符合上述的三阶段抽象计算过程。



#### Self Attention模型



在attention机制中，query是来自外在的张量，而Self Attention的query则是自己本身

但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。除此外，Self Attention对于增加计算的并行性也有直接帮助作用。这是为何Self Attention逐渐被广泛使用的主要原因。







### transformer

完全基于self-attention的模型

![image-20201113233224742](E:\myBlog\source\_posts\image-20201113233224742.png)



### transformer的改进模型







### 预训练模型







### BERT







### BERT的改进模型







### 总结













------



####  总结

Transformer相比LSTM的优点

1. **完全的并行计算**，Transformer的attention和feed-forward，均可以并行计算。而LSTM则依赖上一时刻，必须串行

2. **减少长程依赖**，利用self-attention将每个字之间距离缩短为1，大大缓解了长距离依赖问题

3. **提高网络深度**。由于大大缓解了长程依赖梯度衰减问题，Transformer网络可以很深，基于Transformer的BERT甚至可以做到24层。而LSTM一般只有2层或者4层。网络越深，高阶特征捕获能力越好，模型performance也可以越高。

   

   缺点：

   

1. 文本长度很长时，比如篇章级别，**计算量爆炸**。self-attention的计算量为O(n^2), n为文本长度。

   

2. Transformer参数量较大，在大规模数据集上，效果远好于LSTM。但在**小规模数据集**上，如果不是利用pretrain models，效果不一定有LSTM好。

   



### 🚀Transformer家族2 -- 编码长度优化（Transformer-XL、Longformer）

#### 1 背景



1. Transformer利用self-attention机制进行建模，使得任何两个位置token距离都为1。如果没有内存和算力的限制，Transformer理论上可以编码无限长的文本。但由于attention计算量十分大，而且计算复杂度和序列长度为O(n^2)关系，导致序列长度增加，内存和计算量消耗飞快增加。实际中由于内存和算力有限，一般只能编码一定长度，例如512。

   

为了提升模型的长程编码能力，从而提升模型在长文本，特别是document-level语料上的效果，我们必须对Transformer编码长度进行优化。





#### 2 Transformer-XL

![在这里插入图片描述](E:\myBlog\source\_posts\ctFMhR8IEdPVSZ1.png)
论文信息：2019年01月，谷歌 & CMU，ACL 2019
论文地址 https://arxiv.org/abs/1901.02860
代码和模型地址 https://github.com/kimiyoung/transformer-xl



##### 2.1 为什么需要Transformer-XL

为了解决长文本编码问题，原版Transformer采用了固定编码长度的方案，例如512个token。将长文本按照固定长度，切分为多个segment。每个segment内部单独编码，segment之间不产生交互信息。这种方式的问题如下

1. 模型无法建模超过固定编码长度的文本
2. segment之间没有交互信息，导致了文本碎片化。长语句的编码效果有待提升。
3. predict阶段，decoder每生成一个字，就往后挪一个，没有重复利用之前信息，导致计算量爆炸

train和evaluate过程如下![在这里插入图片描述](E:\myBlog\source\_posts\8YW7R4Q5J6zbmD1.png)



##### 2.2 实现方法

###### 2.2.1 Segment-Level Recurrence with State Reuse 片段级递归和信息复用

Transformer-XL在编码后一个segment时，将前一个segment的隐层缓存下来。后一个segment的self-attention计算，会使用到前一个segment的隐层。后一个segment的第n+1层，对前一个segment的第n层隐层进行融合。故最大编码长度理论上为O(N × L)。

在预测阶段，由于对segment隐层使用了缓存，故每预测一个词，不需要重新对之前的文本进行计算。大大提升了预测速度，最大可达到原始Transformer的1800倍。如下图所示
![在这里插入图片描述](E:\myBlog\source\_posts\KmazXviordxyZuw.png)



###### 2.2.2 Relative Positional Encodings 相对位置编码

segment递归中有个比较大的问题，就是如何区分不同segment中的相同位置。如果采用原版Transformer中的绝对编码方案，两者是无法区分的。如下
![gh](E:\myBlog\source\_posts\38l7XhVjZnqyuBe.png)不同segment中的相同位置，其position encoding会相同。这显然是有问题的。Transformer-XL将绝对位置编码改为了q和k之间的相对位置编码，代表了两个token之间的相对位置。从语义上讲，是make sense的。我们来看看具体实现方式。

绝对位置编码的attention计算如下
![在这里插入图片描述](E:\myBlog\source\_posts\IKCVE7riwaQ9Zl1.png)分为四部分

1. query的token encoding和 key的token encoding，之间的关联信息
2. query的token encoding和 key的position encoding，之间的关联信息。Uj为绝对位置j的编码向量
3. query的position encoding和 key的token encoding，之间的关联信息。Ui为绝对位置i的编码向量
4. query的position encoding和 key的position encoding，之间的关联信息

而采用相对位置编码后，attention计算如下
![在这里插入图片描述](E:\myBlog\source\_posts\Id9fV3bxSnmHyu2.png)同样包含四部分，仍然为二者token encoding和position encoding之间的关联关系。区别在于

1. Ri-j为i和j之间相对位置编码，其中R为相对位置编码矩阵
2. u和v为query的位置编码，采用一个固定向量。因为采用相对位置编码后，无法对单个绝对位置进行编码了。文中称为global content bias，和global positional bias

也有其他文章，采用了不同的相对位置编码方案。比如"Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani.2018. Self-attention with relative position representations. arXiv preprint arXiv:1803.02155." 中只有a和b两部分，丢掉了c和d。Transformer-XL对两种方案进行了对比实验，证明前一种好。



##### 2.3 实验结果

###### 长文本编码效果

![在这里插入图片描述](E:\myBlog\source\_posts\IyoR924J3BQWrOu.png)
在WikiText-103上的实验结果。WikiText-103包含词级别的超长文本，平均每篇文章长度为3.6K token。利用它可以验证模型的长文本编码能力。实验结果表明Transformer-XL large的PPL最低，效果最好。同时作者在One Billion Word、enwik8、text8上均进行了实验，都表明Transformer-XL效果最好。



###### 有效编码长度

![在这里插入图片描述](E:\myBlog\source\_posts\oKQv3akDcXWOZ7B.png)
模型可编码的有效长度如上，r为top-r难度样本上的表现。Transformer-XL比RNN长80%，比Transformer长450%。证明Transformer-XL可编码长度最长，长程捕获能力最强。



###### 预测速度

![在这里插入图片描述](E:\myBlog\source\_posts\g2TyZhWtRSEA7pB.png)
在不同segment长度下，模型预测速度的对比。和原始Transformer相比，预测速度最大可以提升1874倍。



###### 消融分析

文章对片段级递归和相对位置编码两个Method进行了消融分析，如下
![在这里插入图片描述](E:\myBlog\source\_posts\4EeY78JcQojSuhD.png)
两个改进点均有作用，其中片段级递归作用更大。



#### 3 Longformer

![在这里插入图片描述](E:\myBlog\source\_posts\KhcS2VIyvNEAwrg.png)
论文信息：2020年04月，allenai
论文地址 https://arxiv.org/abs/2004.05150
代码和模型地址 https://github.com/allenai/longformer



##### 3.1 改进方法

###### 3.1.1 attention稀疏化

Transformer不能捕获长距离信息，本质原因还是因为计算量过大导致的。那我们通过降低attention计算量，是不是就可以提升长距离编码能力呢。答案是肯定的，LongFormer提出了三种attention稀疏化的方法，来降低计算量。
![在这里插入图片描述](E:\myBlog\source\_posts\hDmpknyqTSIUEW7.png)
a是原始的全连接方式的attention。后面三种为文章使用的稀疏attention。

1. Sliding Window attention。滑窗方式的attention。假设序列长度为n，滑窗大小w，则每个位置的attention只和滑窗范围内的token进行计算，复杂度从O(n^2)变为了O(n * w)。当w << n时，可以大大降低计算量。
2. Dilated Sliding Window attention。受到空洞卷积的启发，提出了空洞滑窗attention。看下面这张图就明白了。

![在这里插入图片描述](E:\myBlog\source\_posts\ZwAy1H3kB7SjIP5.png)

1. Global Attention + sliding window。某些关键位置采用全局attention，这些位置的attention和所有token进行计算。而其他相对不关键的位置则采用滑窗attention。那什么叫关键位置呢？作者举例，分类问题中[CLS]为关键位置，需要计算全局attention。QA中question每个位置均为关键位置，同样计算全局attention。



###### 3.1.2 Tensor Virtual Machine (TVM)

作者使用了TVM构建CUDA kernel，加快了longformer的速度，并降低了显存需求。这个又是另一个模型加速方面的话题，我们就先不展开了。



##### 3.2 实验结果

###### 大小模型效果

![在这里插入图片描述](E:\myBlog\source\_posts\ZThJDNXByQp2Lod.png)作者在大小模型上均实验了LongFormer的效果。小模型为12 layers，512 hidden。大模型为30 layers，512 hidden。在text8和enwik8数据集上。小模型达到了SOTA。大模型比18层的Transformer-XL好，虽然不如Adaptive-span-Transformer和Compressive，但胜在可以pretrain-finetune



###### 消融分析

![在这里插入图片描述](E:\myBlog\source\_posts\pMCoW2ZLv6gq3j4.png)消融分析中，可以发现

1. Dilation空洞，有一定的收益
2. top layer滑窗大小比bottom layer大时，效果好一些。这个也是make sense的。因为top layer捕获高维语义，关联信息距离相对较远，窗口应该尽量大一些。



###### 语料长度

![在这里插入图片描述](E:\myBlog\source\_posts\IjGfH7Mi1LK2hsC.png)
从上表中我们发现，语料都是特别长的长文本。LongFormer真的是document级别的Transformer。



###### 下游任务finetune效果

![在这里插入图片描述](E:\myBlog\source\_posts\JdZUrh6L3QFBHIn.png)
![在这里插入图片描述](E:\myBlog\source\_posts\WlOZR3iaJSA2nIU.png)
第一个table为RoBERTa和LongFormer在问答、指代消解、分类任务中的对比。第二个table为这几个任务数据集平均文本长度。每项任务都是超出RoBERTa，当文本长度大于512时，performance提升特别明显。更加说明了长程捕获能力在NLP中的重要性。

### 🚀Transformer家族3 -- 计算效率优化（Adaptive-Span、Reformer、Lite-Transformer）



#### 1 背景

上文我们从编码长度优化的角度，分析了如何对Transformer进行优化。Transformer-XL、LongFormer等模型，通过片段递归和attention稀疏化等方法，将长文本编码能力提升到了很高的高度。基本已经克服了Transformer长文本捕获能力偏弱的问题，使得下游任务模型performance得到了较大提升，特别是文本较长（大于512）的任务上。

但Transformer计算量和内存消耗过大的问题，还亟待解决。事实上，Transformer-XL、LongFormer已经大大降低了内存和算力消耗。毕竟Transformer之所以长距离编码能力偏弱，就是因为其计算量是序列长度的平方关系，对算力需求过大，导致当前GPU/TPU不能满足需求。编码长度优化和计算量优化，二者是相辅相成的。但着眼于论文的出发点，我们还是分为两个不同的章节进行分析。毕竟总不能所有模型都放在一个章节吧（_）。

本文我们带来Adaptive-Span Transformer、Reformer、Lite-Transformer等几篇文章



#### 2 Adaptive-Span Transformer

![在这里插入图片描述](E:\myBlog\source\_posts\SZIDO7KcoGHqvhF.png)
论文信息：2019年5月，FaceBook，ACL2019
论文地址 https://arxiv.org/pdf/1905.07799.pdf
代码和模型地址 https://github.com/facebookresearch/adaptive-span



##### 2.1 为什么需要Adaptive-Span

之前Transformer-XL将长文本编码能力提升到了较高高度，但是否每个layer的每个head，都需要这么长的attention呢？尽管使用了多种优化手段，长距离attention毕竟还是需要较大的内存和算力。研究发现，大部分head只需要50左右的attention长度，只有少部分head需要较长的attention。这个是make sense的，大部分token只和它附近的token有关联。如下图
![在这里插入图片描述](E:\myBlog\source\_posts\nOstVdScKG68CDz.png)我们是否可以实现attention span长度的自适应呢？让不同的layer的不同的head，自己去学习自己的attention span长度呢？Adaptive-Span Transformer给出了肯定答案。



##### 2.2 实现方案

文章设定每个attention head内的token计算，都使用同一个span长度。我们就可以利用attention mask来实现自适应span。对每个head都添加一个attention mask，mask为0的位置不进行attention计算。文章设计的mask函数如下
![在这里插入图片描述](E:\myBlog\source\_posts\yRVqrjfJPgBKlIk.png)
R为超参，控制曲线平滑度。其为单调递减函数，如下图。
![在这里插入图片描述](E:\myBlog\source\_posts\hvoFmDIp1BjgV5f.png)



##### 2.3 实验结果

![在这里插入图片描述](E:\myBlog\source\_posts\EDsBxfipL4HcOwF.png)
和Transformer家族其他很多模型一样，Adaptive-span也在字符级别的语言模型上进行了验证，数据集为text8。如上，Transformer注意力长度固定为512，结论如下

1. Transformer-XL长程编码能力确实很强，平均span可达3800。
2. 注意力长度确实不需要总那么长，Adaptive-Span大模型上，平均长度只有245
3. Adaptive-Span在算力需求很小（只有XL的1/3）的情况下，效果可以达到SOTA。

![在这里插入图片描述](E:\myBlog\source\_posts\IDjAR2ucfELsegw.png)上面是在enwik8上的结果。Adaptive-Span又一次在算力很小的情况下，达到了最优效果。值得注意的是，64层的Transformer居然需要120G的计算量，又一次证明了原版Transformer是多么的吃计算资源。另外Transformer-XL在节省计算资源上，其实也算可圈可点。



#### 3 Reformer

![在这里插入图片描述](E:\myBlog\source\_posts\G8AVxR7fuCZdPHn.png)
论文信息：2020年1月，谷歌，ICLR2020
论文地址 https://arxiv.org/abs/2001.04451
代码和模型地址 https://github.com/google/trax/tree/master/trax/models/reformer



##### 3.1 为什么需要Reformer

Transformer内存和计算量消耗大的问题，一直以来广为诟病，并导致其一直不能在长文本上进行应用。（BERT、RoBERTa均设置最大长度为512）。Reformer认为Transformer有三大问题

1. attention层计算量和序列长度为平方关系，导致无法进行长距离编码
2. 内存占用和模型层数呈N倍关系，导致加深Transformer层数，消耗的内存特别大
3. feed-forward的dff比隐层dmodel一般大很多，导致FF层占用的内存特别大

针对这几个问题，Reformer创新性的提出了三点改进方案

1. LOCALITY-SENSITIVE HASHING 局部敏感hash，使得计算量从 O(L^2)降低为O(L log L) ,L为序列长度
2. Reversible Transformer 可逆Transformer，使得N层layers内存消耗变为只需要一层，从而使得模型加深不会受内存限制。
3. Feed-forward Chunking 分块全连接，大大降低了feed-forward层的内存消耗。

Reformer是Transformer家族中最为关键的几个模型之一（去掉之一貌似都可以，顶多Transformer-XL不答应），其创新新也特别新颖，很多思想值得我们深入思考和借鉴。其效果也是特别明显，大大提高了内存和计算资源效率，编码长度可达64k。下面针对它的三点改进方案进行分析，有点难懂哦。



##### 3.2 实现方案

###### 3.2.1 LOCALITY-SENSITIVE HASHING 局部敏感hash

局部敏感hash有点难懂，Reformer针对Transformer结构进行了深度灵魂拷问



###### Query和Key必须用两套吗

Transformer主体结构为attention，原版attention计算方法如下
![在这里插入图片描述](E:\myBlog\source\_posts\Z1fYDpO6BnLyHj4.png)
每个token，利用其query向量，和其他token的key向量进行点乘，从而代表两个token之间的相关性。归一化后，利用得到的相关性权重，对每个token的value向量进行加权求和。首先一个问题就是，query和key向量可以是同一套吗？我们可否利用key向量去和其他token的key计算相关性呢？

为此文章进行实验分析，证明是可行的。个人认为这一点也是make sense的。
![在这里插入图片描述](E:\myBlog\source\_posts\YyTZDG1Bohkswit.png)在文本和图像上，Q=K的attention，和普通attention，效果差别不大。



###### 必须和每个token计算相关性吗

原版attention中，一个token必须和序列中其他所有token计算相关性，导致计算量随序列长度呈平方关系增长，大大制约了可编码最大长度。那必须和每个token计算相关性吗？其实之前Adaptive-Span Transformer也深度拷问过这个话题。它得出的结论是，对于大部分layer的multi-head，长度50范围内进行attention就已经足够了。不过Adaptive-Span采取的方法还是简单粗暴了一点，它约定每个head的attention span长度是固定的，并且attention span为当前token附近的其他token。

Adaptive-Span Transformer的这种方法显然还是没有抓住Attention计算冗余的痛点。Attention本质是加权求和，权重为两个token间的相关性。最终结果取决于较大的topk权重，其他权值较小的基本就是炮灰。并且softmax归一化更是加剧了这一点。小者更小，大者更大。为了减少计算冗余，我们可以只对相关性大的其他token的key向量计算Attention。



###### 怎么找到相关性大的向量呢

我们现在要从序列中找到与本token相关性最大的token，也就是当前key向量与哪些key向量相关性大。极端例子，如果两个向量完全相同，他们的相关性是最高的。确定两个高维向量的相关性确实比较困难，好在我们可以利用向量Hash来计算。

Reformer采用了局部敏感hash。我们让两个key向量在随机向量上投影，将它们划分到投影区间内。
![在这里插入图片描述](E:\myBlog\source\_posts\gSHlhWRaoqfk9pN.png)
如图所示，划分了四个区间（4个桶bucket），进行了三次Hash。第一次Hash，使得上面两个向量分别归入bucket0和bucket3中，下面两个向量都归入bucket0。第二次Hash，上面两个向量和下面两个，均归入到bucket2中了。我们可以发现

1. 相似的向量，也就是相关性大的，容易归入到一个bucket中
2. 局部敏感Hash还是有一定的错误率的，我们可以利用多轮Hash来缓解。这也是Reformer的做法，它采取了4轮和8轮的Hash。



###### 整个流程

经过局部敏感Hash后，我们可以将相关性大的key归入同一个bucket中。这样只用在bucket内进行普通Attention即可，大大降低了计算冗余度。为了实现并行计算，考虑到每个bucket包含的向量数目可能不同，实际处理中需要多看一个bucket。整个流程如下
![在这里插入图片描述](E:\myBlog\source\_posts\ZS5dCT8nVcaiob4.png)

1. 让query等于key
2. 局部敏感Hash（LSH）分桶。上图同一颜色的为同一个桶，共4个桶
3. 桶排序，将相同的桶放在一起
4. 为了实现并行计算，将所有桶分块（chunk），每个chunk大小相同
5. 桶内计算Attention，由于之前做了分块操作，所以需要多看一个块。



###### 多轮LSH

为了减少分桶错误率，文章采用了多次分桶，计算LSH Attention，Multi-round LSH attention。可以提升整体准确率。如下表。![在这里插入图片描述](E:\myBlog\source\_posts\L4BDoiwTfaRjInv.png)



###### 3.2.2 REVERSIBLE TRANSFORMER 可逆Transformer

LSH局部敏感Hash确实比较难理解，可逆Transformer相对好懂一些。这个方案是为了解决Transformer内存占用量，随layers层数线性增长的问题。为什么会线性增长呢？原因是反向传播中，梯度会从top layer向bottom layer传播，所以必须保存住每一层的Q K V向量，也就导致N层就需要N套Q K V。

那有没有办法不保存每一层的Q K V呢？可逆Transformer正是这个思路。它利用时间换空间的思想，只保留一层的向量，反向传播时，实时计算出之前层的向量。所以叫做Reversible。Reformer每一层分为两部分，x1和x2。输出也两部分，y1和y2。计算如下

![image-20201028214552153](E:\myBlog\source\_posts\gKZT1mRFhEj48nU.png)



采用可逆残差连接后，模型效果基本没有下降。这也是make sense的，毕竟可逆是从计算角度来解决问题的，对模型本身没有改变。
![在这里插入图片描述](E:\myBlog\source\_posts\hElWr4uHtUomVYG.png)



###### 3.2.3 Feed-Forward chunking FF层分块

针对fead-forward层内存消耗过大的问题，Reformer也给出了解决方案，就是FF层分块。如下
![在这里插入图片描述](E:\myBlog\source\_posts\rpzwVXOSUavEs18.png)



##### 3.3 实验结果

###### 内存和时间复杂度

Reformer三个创新点，大大降低了内存和时间复杂度，消融分析如下
![在这里插入图片描述](E:\myBlog\source\_posts\5ixhnpvKeQHrBw1.png)



###### 模型效果

如下为在机器翻译上的效果。Reformer减少了算力消耗，同时也大大增加了长文本编码能力，故模型效果也得到了提升。如下。
![在这里插入图片描述](E:\myBlog\source\_posts\8EtHVoBsfYxnmhO.png)





#### 4 Lite Transformer

![在这里插入图片描述](E:\myBlog\source\_posts\9UfkunTVEpL5XBG.png)
论文信息：2020年4月，MIT & 上海交大，ICLR2020
论文地址 https://arxiv.org/abs/2004.11886
代码和模型地址 https://github.com/mit-han-lab/lite-transformer



##### 4.1 为什么要做Lite Transformer

主要出发点仍然是Transformer计算量太大，计算冗余过多的问题。跟Adaptive-Span Transformer和Reformer想法一样，Lite Transformer也觉得没必要做Full Attention，很多Attention连接是冗余的。不一样的是，它通过压缩Attention通道的方式实现，将多头减少了一半。与Base Transformer相比，计算量减少了2.5倍。并且文章使用了量化和剪枝技术，使得模型体积减小了18.2倍。



##### 4.2 实现方案

实现方案很简单，仍然采用了原版Transformer的seq2seq结构，创新点为

1. multiHead self-attention变为了两路并行，分别为一半的通道数（多头）。如下图a所示。其中左半部分为正常的fully attention，它用来捕获全局信息。右半部分为CNN卷积，用来捕获布局信息。最终二者通过FFN层融合。这个架构称为LONG-SHORT RANGE ATTENTION (LSRA)，长短期Attention。
2. 为了进一步降低计算量，作者将CNN转变为了一个depth wise卷积和一个线性全连接。dw卷积在mobileNet中有讲过，不清楚可自行谷歌。

![在这里插入图片描述](E:\myBlog\source\_posts\WQtJvpbP9rUczhs.png)



##### 4.3 实验结果

###### 计算复杂度

![在这里插入图片描述](E:\myBlog\source\_posts\w6b1LlqIpZgt8fX.png)
如上图，在文本摘要任务上，Lite Transformer计算量相比Base Transformer，减少了2.5倍。同时Rouge指标基本没变。



###### 模型体积

![在这里插入图片描述](E:\myBlog\source\_posts\JYzeocGBUr6jy4W.png)
Lite Transformer模型体积只有Transformer的2.5分之一，通过8bit量化和剪枝，最终模型体积下降了18.2倍。







### 🚀邱锡鹏教授：NLP预训练模型综述

#### **1.引言**

随深度学习的发展，多种神经网络都被应用在 NLP 任务中，比如 CNN、RNN、GNN 和 attention 机制等，但由于现有的数据集对于大部分有监督 NLP 任务来说都很小，因此，早期的模型对 NLP 任务来说都很“浅”，往往只包含 1-3 层。

而预训练模型（Pre-trained Models, PTMs）的出现将NLP带入一个新的时代，更“深”的模型和训练技巧的增强也使得 PTMs 由“浅”变“深”，在多项任务都达到了 SOTA 性能。



#### **2.背景**

##### **2.1 语言表示学习**

对于语言来说，一个好的表示应当描绘语言的内在规则比如词语含义、句法结构、语义角色甚至语用。

![img](E:\myBlog\source\_posts\ZmF3yXaiHPME1LQ.png)

而分布式表示的核心思想就是通过低维实值向量来描述一段文本的意义，而向量的每一个维度都没有对于意义，整体则代表一个具体的概念。图 1 是 NLP 的通用神经体系架构。

有两种 embedding（词嵌入）方式：上下文嵌入和非上下文嵌入，两者的区别在于词的 embedding 是否根据词出现的上下文动态地改变。

**非上下文嵌入：**表示语言的第一步就是将分离的语言符号映射到分布式嵌入空间中。也就是对于词汇表中的每个单词（词根），通过 lookup table 映射到一个向量。

这种嵌入方式有两个局限：一是一个词通过这种方法获得的词嵌入总是静态且与上下文无关的，无法处理多义词；二是难以解决不在词汇表中的词（针对这个问题，很多 NLP 任务提出了字符级或词根级的词表示，如 CharCNN、FastText、Byte-Pair Encoding (BPE)）。

**上下文嵌入：**为解决多义性和上下文相关的问题，将词在不同上下文的语义做区分。通过对词（词根）的 token 加一层 Neural Contextual Encoder（神经上下文编码器）得到词的上下文嵌入。



有了上下文嵌入，就可以代入到具体的NLP任务中



以下是NLP领域主要模型的发展

随着BERT的出现，模型的规模越来越大，参数越来越多，训练的无标签数据集越来越大。因为随着参数增大，模型的性能并没有出现饱和的状态。GPT3有1750亿个参数。

![image-20201114154032790](E:\myBlog\source\_posts\image-20201114154032790.png)



在问答任务AQuAD 2.0 中，预训练模型精度已经和人类差不多了

![image-20201114164427442](E:\myBlog\source\_posts\image-20201114164427442.png)



felf-attention 机制 ，得到“the”的上下文表示

![image-20201114154601028](E:\myBlog\source\_posts\image-20201114154601028.png)



##### **2.3 为什么要预训练？**



模型参数的数量增长迅速，而为了训练这些参数，就需要更大的数据集来避免过拟合，但是标注的数据集太少，大规模的标注数据集成本非常高，很多时候需要专家来进行标注。 而相比之下，大规模未标注的语料却很容易构建。



为了利用大量的未标注文本数据，我们可以先从其中学习一个好的表示，再将这些表示用在别的任务中。这一通过 PTMs 从未标注大规模数据集中提取表示的预训练过程在很多 NLP 任务中都取得了很好的表现。

预训练的优点可以总结为以下三点：1 在大规模语料上通过预训练学习通用语言表示对下游任务很有帮助，更好地泛化到不同的任务；2) 预训练提供了更好的模型初始化参数，使得在目标任务上有更好的泛化性能和更快的收敛速度；3) 预训练是一种有效的正则化方法，能够避免复杂模型在小数据集上过拟合。



##### 预训练任务汇总

自监督学习：去⼈为地构造⼀些任务，这些 任务在实际中可能不存在 , 但是可以通过这些任务, 去学习到⼀些隐含 的知识



![image-20201114160905548](E:\myBlog\source\_posts\image-20201114160905548.png)









#### ELMo模型



![image-20201114202903029](E:\myBlog\source\_posts\image-20201114202903029.png)



ELMo的预训练过程











##### **3.3 如何压缩PTMs**

预训练模型往往包含至少几千万个参数，这也使得模型难以部署到生活中的线上服务以及资源有限的设备上，这就使得模型压缩成为一条可能能够压缩模型尺寸并提高计算效率的方法。



压缩 PTMs 一般有四个方法：

- **剪枝（pruning）：**去除不那么重要的参数（e.g. 权重、层数、通道数、attention heads）

- **量化（weight quantization）：**使用占位更少（低精度）的参数

- **参数共享（parameter sharing）：**相似模型单元间共享参数

- **知识蒸馏（knowledge diistillation）：**用一些优化目标从大型 teacher 模型学习一个小的 student 模型，一些利用知识蒸馏的 PTMs 见表 3

  

![img](E:\myBlog\source\_posts\5tdJqBHve3XYuCo.png)









#### **4.如何将PTMs应用至下游任务**         



![image-20201114165642819](E:\myBlog\source\_posts\image-20201114165642819.png)

预训练完模型数据集不要了，只保留模型以及参数，迁移到下游任务，在目标数据集上进行微调









#### **7.未来方向**

##### **7.1 PTMs的上界**

随 BERT 的出现，我们可以发现，很多模型都可以通过更长的训练步长不在和更大的语料来提升性能



PTMs 的共同目标都是学习语言的本质通用知识(或者说是世界的知识)，然而，随着模型的不断加深，语料的不断增大，训练模型的花销也越来越大。一种更可行的解决方案是设计更有效的模型架构、自监督预训练任务、优化器和软硬件方面的技巧等。



##### **7.2 面向任务的预训练与模型压缩**

任务定向

模型精简 

在实践中，不同的下游任务要求 PTMs 拥有不同的功能。而 PTMs 与下游目标任务间的差异通常表现在两方面：模型架构与数据分布。较大的 PTMs 通常情况下会有更好的性能，但实际问题是如何在低容量设备和低时延应用上使用如此庞大的 PTM。

除此之外，我们可以通过模型压缩来将通用 PTMs 教给面向对象的 PTM。尽管 CV 中对 CNNs 的压缩已经非常成熟，但 Tansformer 的全连接结构使得模型压缩非常具有挑战性。







------



BERT模型从模型创新角度看一般，创新不算大。但是效果太好了，基本刷新了很多NLP的任务的最好性能，有些任务还被刷爆了，这个才是关键。

另外一点是Bert具备广泛的通用性，就是说绝大部分NLP任务都可以采用类似的两阶段模式直接去提升效果，这个第二关键。客观的说，把Bert当做最近两年NLP重大进展的集大成者更符合事实。





预训练最初是应用在图像领域的



模型中大量参数通过大的训练集合比如 ImageNet预先训练好直接拿来初始化大部分网络结构参数，然后再用下游任务上 Fine-tuning过程去调整参数让它们更适合解决下游任务。

CNN结构来说，不同层级的神经元学习到了不同类型的图像特征，由底向上特征形成层级结构预训练好的网络参数，尤其是底层的网络参数抽取出特征跟具体任务越无关，越具备任务的通用性，所以这是为何一般用底层预训练好的参数初始化新任务 网络参数的原因



### NLP领域的预训练模型

#### Word Embedding考古史 





![img](E:\myBlog\source\_posts\4aFBLDCEQmgXp9Z.jpg)



什么是语言模型？其实看上面这张PPT上扣下来的图就明白了，为了能够量化地衡量哪个句子更像一句人话，可以设计如上图所示函数，核心函数P的思想是根据句子里面前面的一系列前导单词预测后面跟哪个单词的概率大小（理论上除了上文之外，也可以引入单词的下文联合起来预测单词出现概率）。句子里面每个单词都有个根据上文预测自己的过程，把所有这些单词的产生概率乘起来，数值越大代表这越像一句人话。语言模型压下暂且不表，我隐约预感到我这么讲你可能还是不太会明白，但是大概这个意思，不懂的可以去网上找，资料多得一样地汗牛冲动。

假设现在让你设计一个神经网络结构，去做这个语言模型的任务，就是说给你很多语料做这个事情，训练好一个神经网络，训练好之后，以后输入一句话的前面几个单词，要求这个网络输出后面紧跟的单词应该是哪个，你会怎么做？



![img](E:\myBlog\source\_posts\SUI2jF7xEsaz9Z4.jpg)



你可以像上图这么设计这个网络结构，这其实就是大名鼎鼎的中文人称“神经网络语言模型”，英文小名NNLM的网络结构，用来做语言模型。这个工作有年头了，是个陈年老工作，是Bengio 在2003年发表在JMLR上的论文。它生于2003，火于2013，以后是否会不朽暂且不知，但是不幸的是出生后应该没有引起太大反响，沉寂十年终于时来运转沉冤得雪，在2013年又被NLP考古工作者从海底湿淋淋地捞出来了祭入神殿。为什么会发生这种技术奇遇记？你要想想2013年是什么年头，是深度学习开始渗透NLP领域的光辉时刻，万里长征第一步，而NNLM可以算是南昌起义第一枪。在深度学习火起来之前，极少有人用神经网络做NLP问题，如果你10年前坚持用神经网络做NLP，估计别人会认为你这人神经有问题。所谓红尘滚滚，谁也挡不住历史发展趋势的车轮，这就是个很好的例子。

上面是闲话，闲言碎语不要讲，我们回来讲一讲NNLM的思路。先说训练过程，现在看其实很简单，见过RNN、LSTM、CNN后的你们回头再看这个网络甚至显得有些简陋。学习任务是输入某个句中单词Wt=“Bert”前面句子的t-1个单词，要求网络正确预测单词Bert，即最大化：

<img src="https://i.loli.net/2020/10/30/k8XpuUwT47WOo5Q.png" alt="image-20201030202319906" style="zoom:50%;" />

前面任意单词Wi用Onehot编码（比如：0001000）作为原始单词输入，之后乘以矩阵Q后获得向量C（Wi），每个单词的 C（Wi）拼接，上接隐层，然后接softmax去预测后面应该后续接哪个单词。这个 C（Wi）是什么？这其实就是单词对应的Word Embedding值，那个矩阵Q包含V行，V代表词典大小，每一行内容代表对应单词的Word embedding值。只不过Q的内容也是网络参数，需要学习获得，训练刚开始用随机值初始化矩阵Q，当这个网络训练好之后，矩阵Q的内容被正确赋值，每一行代表一个单词对应的Word embedding值。所以你看，通过这个网络学习语言模型任务，这个网络不仅自己能够根据上文预测后接单词是什么，同时获得一个副产品，就是那个矩阵Q，这就是单词的Word Embedding是被如何学会的。

2013年最火的用语言模型做Word Embedding的工具是Word2Vec，后来又出了Glove，Word2Vec是怎么工作的呢？看下图。



![img](E:\myBlog\source\_posts\bXq5RvhS3niaTxB.jpg)



Word2Vec的网络结构其实和NNLM是基本类似的，只是这个图长得清晰度差了点，看上去不像，其实它们是亲兄弟。不过这里需要指出：尽管网络结构相近，而且也是做语言模型任务，但是其训练方法不太一样。Word2Vec有两种训练方法，一种叫CBOW，核心思想是从一个句子里面把一个词抠掉，用这个词的上文和下文去预测被抠掉的这个词；第二种叫做Skip-gram，和CBOW正好反过来，输入某个单词，要求网络预测它的上下文单词。而你回头看看，NNLM是怎么训练的？是输入一个单词的上文，去预测这个单词。这是有显著差异的。为什么Word2Vec这么处理？原因很简单，因为Word2Vec和NNLM不一样，NNLM的主要任务是要学习一个解决语言模型任务的网络结构，语言模型就是要看到上文预测下文，而word embedding只是无心插柳的一个副产品。但是Word2Vec目标不一样，它单纯就是要word embedding的，这是主产品，所以它完全可以随性地这么去训练网络。

为什么要讲Word2Vec呢？这里主要是要引出CBOW的训练方法，BERT其实跟它有关系，后面会讲它们之间是如何的关系，当然它们的关系BERT作者没说，是我猜的，至于我猜的对不对，后面你看后自己判断。



![img](E:\myBlog\source\_posts\aM9lpsNOS7vrmnq.jpg)

使用Word2Vec或者Glove，通过做语言模型任务，就可以获得每个单词的Word Embedding，那么这种方法的效果如何呢？上图给了网上找的几个例子，可以看出有些例子效果还是很不错的，一个单词表达成Word Embedding后，很容易找出语义相近的其它词汇。

我们的主题是预训练，那么问题是Word Embedding这种做法能算是预训练吗？这其实就是标准的预训练过程。要理解这一点要看看学会Word Embedding后下游任务是怎么用它的。



![img](E:\myBlog\source\_posts\YEf95lnIW28OGRa.jpg)

假设如上图所示，我们有个NLP的下游任务，比如QA，就是问答问题，所谓问答问题，指的是给定一个问题X，给定另外一个句子Y,要判断句子Y是否是问题X的正确答案。问答问题假设设计的网络结构如上图所示，这里不展开讲了，懂得自然懂，不懂的也没关系，因为这点对于本文主旨来说不关键，关键是网络如何使用训练好的Word Embedding的。它的使用方法其实和前面讲的NNLM是一样的，句子中每个单词以Onehot形式作为输入，然后乘以学好的Word Embedding矩阵Q，就直接取出单词对应的Word Embedding了。这乍看上去好像是个查表操作，不像是预训练的做法是吧？其实不然，那个Word Embedding矩阵Q其实就是网络Onehot层到embedding层映射的网络参数矩阵。所以你看到了，使用Word Embedding等价于什么？等价于把Onehot层到embedding层的网络用预训练好的参数矩阵Q初始化了。这跟前面讲的图像领域的低层预训练过程其实是一样的，区别无非Word Embedding只能初始化第一层网络参数，再高层的参数就无能为力了。下游NLP任务在使用Word Embedding的时候也类似图像有两种做法，一种是Frozen，就是Word Embedding那层网络参数固定不动；另外一种是Fine-Tuning，就是Word Embedding这层参数使用新的训练集合训练也需要跟着训练过程更新掉。

上面这种做法就是18年之前NLP领域里面采用预训练的典型做法，之前说过，Word Embedding其实对于很多下游NLP任务是有帮助的，只是帮助没有大到闪瞎忘记戴墨镜的围观群众的双眼而已。那么新问题来了，为什么这样训练及使用Word Embedding的效果没有期待中那么好呢？答案很简单，因为Word Embedding有问题呗。这貌似是个比较弱智的答案，关键是Word Embedding存在什么问题？这其实是个好问题。



![img](E:\myBlog\source\_posts\U3YwNJ1RmydDukM.jpg)



这片在Word Embedding头上笼罩了好几年的乌云是什么？是多义词问题。我们知道，多义词是自然语言中经常出现的现象，也是语言灵活性和高效性的一种体现。多义词对Word Embedding来说有什么负面影响？如上图所示，比如多义词Bank，有两个常用含义，但是Word Embedding在对bank这个单词进行编码的时候，是区分不开这两个含义的，因为它们尽管上下文环境中出现的单词不同，但是在用语言模型训练的时候，不论什么上下文的句子经过word2vec，都是预测相同的单词bank，而同一个单词占的是同一行的参数空间，这导致两种不同的上下文信息都会编码到相同的word embedding空间里去。所以word embedding无法区分多义词的不同语义，这就是它的一个比较严重的问题。

你可能觉得自己很聪明，说这可以解决啊，确实也有很多研究人员提出很多方法试图解决这个问题，但是从今天往回看，这些方法看上去都成本太高或者太繁琐了，有没有简单优美的解决方案呢？

ELMO提供了一种简洁优雅的解决方案。



### 从Word Embedding到ELMO

ELMO是“Embedding from Language Models”的简称，其实这个名字并没有反应它的本质思想，提出ELMO的论文题目：“Deep contextualized word representation”更能体现其精髓，而精髓在哪里？在deep contextualized这个短语，一个是deep，一个是context，其中context更关键。在此之前的Word Embedding本质上是个静态的方式，所谓静态指的是训练好之后每个单词的表达就固定住了，以后使用的时候，不论新句子上下文单词是什么，这个单词的Word Embedding不会跟着上下文场景的变化而改变，所以对于比如Bank这个词，它事先学好的Word Embedding中混合了几种语义 ，在应用中来了个新句子，即使从上下文中（比如句子包含money等词）明显可以看出它代表的是“银行”的含义，但是对应的Word Embedding内容也不会变，它还是混合了多种语义。这是为何说它是静态的，这也是问题所在。ELMO的本质思想是：我事先用语言模型学好一个单词的Word Embedding，此时多义词无法区分，不过这没关系。在我实际使用Word Embedding的时候，单词已经具备了特定的上下文了，这个时候我可以根据上下文单词的语义去调整单词的Word Embedding表示，这样经过调整后的Word Embedding更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以ELMO本身是个根据当前上下文对Word Embedding动态调整的思路。

![img](E:\myBlog\source\_posts\m6NvFoRGhWbji83.jpg)

ELMO采用了典型的两阶段过程，第一个阶段是利用语言模型进行预训练；第二个阶段是在做下游任务时，从预训练网络中提取对应单词的网络各层的Word Embedding作为新特征补充到下游任务中。上图展示的是其预训练过程，它的网络结构采用了双层双向LSTM，目前语言模型训练的任务目标是根据单词 Wi 的上下文去正确预测单词  Wi  ， Wi 之前的单词序列Context-before称为上文，之后的单词序列Context-after称为下文。图中左端的前向双层LSTM代表正方向编码器，输入的是从左到右顺序的除了预测单词外  Wi  的上文Context-before；右端的逆向双层LSTM代表反方向编码器，输入的是从右到左的逆序的句子下文Context-after；每个编码器的深度都是两层LSTM叠加。这个网络结构其实在NLP中是很常用的。使用这个网络结构利用大量语料做语言模型任务就能预先训练好这个网络，如果训练好这个网络后，输入一个新句子 Snew ，句子中每个单词都能得到对应的三个Embedding:最底层是单词的Word Embedding，往上走是第一层双向LSTM中对应单词位置的Embedding，这层编码单词的句法信息更多一些；再往上走是第二层LSTM中对应单词位置的Embedding，这层编码单词的语义信息更多一些。也就是说，ELMO的预训练过程不仅仅学会单词的Word Embedding，还学会了一个双层双向的LSTM网络结构，而这两者后面都有用。



![img](E:\myBlog\source\_posts\ymSXKwFh8WBcEd5.jpg)



上面介绍的是ELMO的第一阶段：预训练阶段。那么预训练好网络结构后，如何给下游任务使用呢？上图展示了下游任务的使用过程，比如我们的下游任务仍然是QA问题，此时对于问句X，我们可以先将句子X作为预训练好的ELMO网络的输入，这样句子X中每个单词在ELMO网络中都能获得对应的三个Embedding，之后给予这三个Embedding中的每一个Embedding一个权重a，这个权重可以学习得来，根据各自权重累加求和，将三个Embedding整合成一个。然后将整合后的这个Embedding作为X句在自己任务的那个网络结构中对应单词的输入，以此作为补充的新特征给下游任务使用。对于上图所示下游任务QA中的回答句子Y来说也是如此处理。因为ELMO给下游提供的是每个单词的特征形式，所以这一类预训练的方法被称为“Feature-based Pre-Training”。至于为何这么做能够达到区分多义词的效果，你可以想一想，其实比较容易想明白原因。



![img](E:\myBlog\source\_posts\b8ToQxv5BPgELI1.jpg)



上面这个图是TagLM采用类似ELMO的思路做命名实体识别任务的过程，其步骤基本如上述ELMO的思路，所以此处不展开说了。TagLM的论文发表在2017年的ACL会议上，作者就是AllenAI里做ELMO的那些人，所以可以将TagLM看做ELMO的一个前导工作。前几天这个PPT发出去后有人质疑说FastAI的在18年4月提出的ULMFiT才是抛弃传统Word Embedding引入新模式的开山之作，我深不以为然。首先TagLM出现的更早而且模式基本就是ELMO的思路；另外ULMFiT使用的是三阶段模式，在通用语言模型训练之后，加入了一个领域语言模型预训练过程，而且论文重点工作在这块，方法还相对比较繁杂，这并不是一个特别好的主意，因为领域语言模型的限制是它的规模往往不可能特别大，精力放在这里不太合适，放在通用语言模型上感觉更合理；再者，尽管ULFMiT实验做了6个任务，但是都集中在分类问题相对比较窄，不如ELMO验证的问题领域广，我觉得这就是因为第二步那个领域语言模型带来的限制。所以综合看，尽管ULFMiT也是个不错的工作，但是重要性跟ELMO比至少还是要差一档，当然这是我个人看法。每个人的学术审美口味不同，我个人一直比较赞赏要么简洁有效体现问题本质要么思想特别游离现有框架脑洞开得异常大的工作，所以ULFMiT我看论文的时候就感觉看着有点难受，觉得这工作没抓住重点而且特别麻烦，但是看ELMO论文感觉就赏心悦目，觉得思路特别清晰顺畅，看完暗暗点赞，心里说这样的文章获得NAACL2018最佳论文当之无愧，比ACL很多最佳论文也好得不是一点半点，这就是好工作带给一个有经验人士的一种在读论文时候就能产生的本能的感觉，也就是所谓的这道菜对上了食客的审美口味。







![img](E:\myBlog\source\_posts\6y7VvCDm9NRpHJx.jpg)



前面我们提到静态Word Embedding无法解决多义词的问题，那么ELMO引入上下文动态调整单词的embedding后多义词问题解决了吗？解决了，而且比我们期待的解决得还要好。上图给了个例子，对于Glove训练出的Word Embedding来说，多义词比如play，根据它的embedding找出的最接近的其它单词大多数集中在体育领域，这很明显是因为训练数据中包含play的句子中体育领域的数量明显占优导致；而使用ELMO，根据上下文动态调整后的embedding不仅能够找出对应的“演出”的相同语义的句子，而且还可以保证找出的句子中的play对应的词性也是相同的，这是超出期待之处。之所以会这样，是因为我们上面提到过，第一层LSTM编码了很多句法信息，这在这里起到了重要作用。



![img](E:\myBlog\source\_posts\YFAVkuIxemlaHPN.jpg)



ELMO经过这般操作，效果如何呢？实验效果见上图，6个NLP任务中性能都有幅度不同的提升，最高的提升达到25%左右，而且这6个任务的覆盖范围比较广，包含句子语义关系判断，分类任务，阅读理解等多个领域，这说明其适用范围是非常广的，普适性强，这是一个非常好的优点。

![img](E:\myBlog\source\_posts\LpMSFe5kX7Qxroh.jpg)



那么站在现在这个时间节点看，ELMO有什么值得改进的缺点呢？首先，一个非常明显的缺点在特征抽取器选择方面，ELMO使用了LSTM而不是新贵Transformer，Transformer是谷歌在17年做机器翻译任务的“Attention is all you need”的论文中提出的，引起了相当大的反响，很多研究已经证明了Transformer提取特征的能力是要远强于LSTM的。如果ELMO采取Transformer作为特征提取器，那么估计Bert的反响远不如现在的这种火爆场面。另外一点，ELMO采取双向拼接这种融合特征的能力可能比Bert一体化的融合特征方式弱，但是，这只是一种从道理推断产生的怀疑，目前并没有具体实验说明这一点。

我们如果把ELMO这种预训练方法和图像领域的预训练方法对比，发现两者模式看上去还是有很大差异的。除了以ELMO为代表的这种基于特征融合的预训练方法外，NLP里还有一种典型做法，这种做法和图像领域的方式就是看上去一致的了，一般将这种方法称为“基于Fine-tuning的模式”，而GPT就是这一模式的典型开创者。



### 从Word Embedding到GPT

![img](E:\myBlog\source\_posts\IDl2hH8j3JVdx6F.jpg)



GPT是“Generative Pre-Training”的简称，从名字看其含义是指的生成式的预训练。GPT也采用两阶段过程，第一个阶段是利用语言模型进行预训练，第二阶段通过Fine-tuning的模式解决下游任务。上图展示了GPT的预训练过程，其实和ELMO是类似的，主要不同在于两点：首先，特征抽取器不是用的RNN，而是用的Transformer，上面提到过它的特征抽取能力要强于RNN，这个选择很明显是很明智的；其次，GPT的预训练虽然仍然是以语言模型作为目标任务，但是采用的是单向的语言模型，所谓“单向”的含义是指：语言模型训练的任务目标是根据 Wi 单词的上下文去正确预测单词 Wi  ， Wi  之前的单词序列Context-before称为上文，之后的单词序列Context-after称为下文。ELMO在做语言模型预训练的时候，预测单词 Wi  同时使用了上文和下文，而GPT则只采用Context-before这个单词的上文来进行预测，而抛开了下文。这个选择现在看不是个太好的选择，原因很简单，它没有把单词的下文融合进来，这限制了其在更多应用场景的效果，比如阅读理解这种任务，在做任务的时候是可以允许同时看到上文和下文一起做决策的。如果预训练时候不把单词的下文嵌入到Word Embedding中，是很吃亏的，白白丢掉了很多信息。

这里强行插入一段简单提下Transformer，尽管上面提到了，但是说的还不完整，补充两句。首先，Transformer是个叠加的“自注意力机制（Self Attention）”构成的深度网络，是目前NLP里最强的特征提取器，注意力这个机制在此被发扬光大，从任务的配角不断抢戏，直到Transformer一跃成为踢开RNN和CNN传统特征提取器，荣升头牌，大红大紫。注意力机制可以参考“[深度学习中的注意力模型](https://zhuanlan.zhihu.com/p/37601161)”，补充下相关基础知识，如果不了解注意力机制你肯定会落后时代的发展。而介绍Transformer比较好的文章可以参考以下两篇文章：一个是Jay Alammar可视化地介绍Transformer的博客文章[The Illustrated Transformer](https://link.zhihu.com/?target=https%3A//jalammar.github.io/illustrated-transformer/) ，非常容易理解整个机制，建议先从这篇看起；然后可以参考哈佛大学NLP研究组写的“[The Annotated Transformer.](https://link.zhihu.com/?target=http%3A//nlp.seas.harvard.edu/2018/04/03/attention.html) ”，代码原理双管齐下，讲得非常清楚。我相信上面两个文章足以让你了解Transformer了，所以这里不展开介绍。

其次，我的判断是Transformer在未来会逐渐替代掉RNN成为主流的NLP工具，RNN一直受困于其并行计算能力，这是因为它本身结构的序列性依赖导致的，尽管很多人在试图通过修正RNN结构来修正这一点，但是我不看好这种模式，因为给马车换轮胎不如把它升级到汽车，这个道理很好懂，更何况目前汽车的雏形已经出现了，干嘛还要执着在换轮胎这个事情呢？是吧？再说CNN，CNN在NLP里一直没有形成主流，CNN的最大优点是易于做并行计算，所以速度快，但是在捕获NLP的序列关系尤其是长距离特征方面天然有缺陷，不是做不到而是做不好，目前也有很多改进模型，但是特别成功的不多。综合各方面情况，很明显Transformer同时具备并行性好，又适合捕获长距离特征，没有理由不在赛跑比赛中跑不过RNN和CNN。

好了，题外话结束，我们再回到主题，接着说GPT。上面讲的是GPT如何进行第一阶段的预训练，那么假设预训练好了网络模型，后面下游任务怎么用？它有自己的个性，和ELMO的方式大有不同。



![img](E:\myBlog\source\_posts\3Gr9vqoPHkSfcAg.jpg)

上图展示了GPT在第二阶段如何使用。首先，对于不同的下游任务来说，本来你可以任意设计自己的网络结构，现在不行了，你要向GPT的网络结构看齐，把任务的网络结构改造成和GPT的网络结构是一样的。然后，在做下游任务的时候，利用第一步预训练好的参数初始化GPT的网络结构，这样通过预训练学到的语言学知识就被引入到你手头的任务里来了，这是个非常好的事情。再次，你可以用手头的任务去训练这个网络，对网络参数进行Fine-tuning，使得这个网络更适合解决手头的问题。就是这样。看到了么？这有没有让你想起最开始提到的图像领域如何做预训练的过程（请参考上图那句非常容易暴露年龄的歌词）？对，这跟那个模式是一模一样的。

这里引入了一个新问题：对于NLP各种花样的不同任务，怎么改造才能靠近GPT的网络结构呢？





![img](E:\myBlog\source\_posts\iJqb8TYLwCvdSVk.jpg)

GPT论文给了一个改造施工图如上，其实也很简单：对于分类问题，不用怎么动，加上一个起始和终结符号即可；对于句子关系判断问题，比如Entailment，两个句子中间再加个分隔符即可；对文本相似性判断问题，把两个句子顺序颠倒下做出两个输入即可，这是为了告诉模型句子顺序不重要；对于多项选择问题，则多路输入，每一路把文章和答案选项拼接作为输入即可。从上图可看出，这种改造还是很方便的，不同任务只需要在输入部分施工即可。

![img](E:\myBlog\source\_posts\qnLcVGo5IK6riYh.jpg)



GPT的效果是非常令人惊艳的，在12个任务里，9个达到了最好的效果，有些任务性能提升非常明显。





![img](E:\myBlog\source\_posts\96zdAXvcOmJTuP2.jpg)

那么站在现在的时间节点看，GPT有什么值得改进的地方呢？其实最主要的就是那个单向语言模型，如果改造成双向的语言模型任务估计也没有Bert太多事了。当然，即使如此GPT也是非常非常好的一个工作，跟Bert比，其作者炒作能力亟待提升。

### Bert的诞生



![img](E:\myBlog\source\_posts\rSJAqOMB4sathDg.jpg)



我们经过跋山涉水，终于到了目的地Bert模型了。

Bert采用和GPT完全相同的两阶段模型，首先是语言模型预训练；其次是使用Fine-Tuning模式解决下游任务。和GPT的最主要不同在于在预训练阶段采用了类似ELMO的双向语言模型，当然另外一点是语言模型的数据规模要比GPT大。所以这里Bert的预训练过程不必多讲了。





![img](E:\myBlog\source\_posts\61JpWKSZ5fF3tNk.jpg)



第二阶段，Fine-Tuning阶段，这个阶段的做法和GPT是一样的。当然，它也面临着下游任务网络结构改造的问题，在改造任务方面Bert和GPT有些不同，下面简单介绍一下。



![img](E:\myBlog\source\_posts\UTQdhtVA7PzcIlF.jpg)



在介绍Bert如何改造下游任务之前，先大致说下NLP的几类问题，说这个是为了强调Bert的普适性有多强。通常而言，绝大部分NLP问题可以归入上图所示的四类任务中：一类是序列标注，这是最典型的NLP任务，比如中文分词，词性标注，命名实体识别，语义角色标注等都可以归入这一类问题，它的特点是句子中每个单词要求模型根据上下文都要给出一个分类类别。第二类是分类任务，比如我们常见的文本分类，情感计算等都可以归入这一类。它的特点是不管文章有多长，总体给出一个分类类别即可。第三类任务是句子关系判断，比如Entailment，QA，语义改写，自然语言推理等任务都是这个模式，它的特点是给定两个句子，模型判断出两个句子是否具备某种语义关系；第四类是生成式任务，比如机器翻译，文本摘要，写诗造句，看图说话等都属于这一类。它的特点是输入文本内容后，需要自主生成另外一段文字。





![img](E:\myBlog\source\_posts\mxJybVWl2OatfUc.jpg)



对于种类如此繁多而且各具特点的下游NLP任务，Bert如何改造输入输出部分使得大部分NLP任务都可以使用Bert预训练好的模型参数呢？上图给出示例，对于句子关系类任务，很简单，和GPT类似，加上一个起始和终结符号，句子之间加个分隔符即可。对于输出来说，把第一个起始符号对应的Transformer最后一层位置上面串接一个softmax分类层即可。对于分类问题，与GPT一样，只需要增加起始和终结符号，输出部分和句子关系判断任务类似改造；对于序列标注问题，输入部分和单句分类是一样的，只需要输出部分Transformer最后一层每个单词对应位置都进行分类即可。从这里可以看出，上面列出的NLP四大任务里面，除了生成类任务外，Bert其它都覆盖到了，而且改造起来很简单直观。尽管Bert论文没有提，但是稍微动动脑子就可以想到，其实对于机器翻译或者文本摘要，聊天机器人这种生成式任务，同样可以稍作改造即可引入Bert的预训练成果。只需要附着在S2S结构上，encoder部分是个深度Transformer结构，decoder部分也是个深度Transformer结构。根据任务选择不同的预训练数据初始化encoder和decoder即可。这是相当直观的一种改造方法。当然，也可以更简单一点，比如直接在单个Transformer结构上加装隐层产生输出也是可以的。不论如何，从这里可以看出，NLP四大类任务都可以比较方便地改造成Bert能够接受的方式。这其实是Bert的非常大的优点，这意味着它几乎可以做任何NLP的下游任务，具备普适性，这是很强的。



![img](E:\myBlog\source\_posts\e7tSMGZjDHmY1Ck.jpg)



Bert采用这种两阶段方式解决各种NLP任务效果如何？在11个各种类型的NLP任务中达到目前最好的效果，某些任务性能有极大的提升。一个新模型好不好，效果才是王道。





![img](E:\myBlog\source\_posts\RniS8uQhpDmcHN6.jpg)

到这里我们可以再梳理下几个模型之间的演进关系。从上图可见，Bert其实和ELMO及GPT存在千丝万缕的关系，比如如果我们把GPT预训练阶段换成双向语言模型，那么就得到了Bert；而如果我们把ELMO的特征抽取器换成Transformer，那么我们也会得到Bert。所以你可以看出：Bert最关键两点，一点是特征抽取器采用Transformer；第二点是预训练的时候采用双向语言模型。

那么新问题来了：对于Transformer来说，怎么才能在这个结构上做双向语言模型任务呢？乍一看上去好像不太好搞。我觉得吧，其实有一种很直观的思路，怎么办？看看ELMO的网络结构图，只需要把两个LSTM替换成两个Transformer，一个负责正向，一个负责反向特征提取，其实应该就可以。当然这是我自己的改造，Bert没这么做。那么Bert是怎么做的呢？我们前面不是提过Word2Vec吗？我前面肯定不是漫无目的地提到它，提它是为了在这里引出那个CBOW训练方法，所谓写作时候埋伏笔的“草蛇灰线，伏脉千里”，大概就是这个意思吧？前面提到了CBOW方法，它的核心思想是：在做语言模型任务的时候，我把要预测的单词抠掉，然后根据它的上文Context-Before和下文Context-after去预测单词。其实Bert怎么做的？Bert就是这么做的。从这里可以看到方法间的继承关系。当然Bert作者没提Word2Vec及CBOW方法，这是我的判断，Bert作者说是受到完形填空任务的启发，这也很可能，但是我觉得他们要是没想到过CBOW估计是不太可能的。

从这里可以看出，在文章开始我说过Bert在模型方面其实没有太大创新，更像一个最近几年NLP重要技术的集大成者，原因在于此，当然我不确定你怎么看，是否认同这种看法，而且我也不关心你怎么看。其实Bert本身的效果好和普适性强才是最大的亮点。



![img](E:\myBlog\source\_posts\LO9j7cIxEJCe2Ay.jpg)

那么Bert本身在模型和方法角度有什么创新呢？就是论文中指出的Masked 语言模型和Next Sentence Prediction。而Masked语言模型上面讲了，本质思想其实是CBOW，但是细节方面有改进。



![img](E:\myBlog\source\_posts\75DVNACdHgtRbS9.jpg)



Masked双向语言模型向上图展示这么做：随机选择语料中15%的单词，把它抠掉，也就是用[Mask]掩码代替原始单词，然后要求模型去正确预测被抠掉的单词。但是这里有个问题：训练过程大量看到[mask]标记，但是真正后面用的时候是不会有这个标记的，这会引导模型认为输出是针对[mask]这个标记的，但是实际使用又见不到这个标记，这自然会有问题。为了避免这个问题，Bert改造了一下，15%的被上天选中要执行[mask]替身这项光荣任务的单词中，只有80%真正被替换成[mask]标记，10%被狸猫换太子随机替换成另外一个单词，10%情况这个单词还待在原地不做改动。这就是Masked双向语音模型的具体做法。



![img](E:\myBlog\source\_posts\MTCajrZPKuF51Ne.jpg)



至于说“Next Sentence Prediction”，指的是做语言模型预训练的时候，分两种情况选择两个句子，一种是选择语料中真正顺序相连的两个句子；另外一种是第二个句子从语料库中抛色子，随机选择一个拼到第一个句子后面。我们要求模型除了做上述的Masked语言模型任务外，附带再做个句子关系预测，判断第二个句子是不是真的是第一个句子的后续句子。之所以这么做，是考虑到很多NLP任务是句子关系判断任务，单词预测粒度的训练到不了句子关系这个层级，增加这个任务有助于下游句子关系判断任务。所以可以看到，它的预训练是个多任务过程。这也是Bert的一个创新。



![img](E:\myBlog\source\_posts\XQ17TqJcYpPoA3i.jpg)

上面这个图给出了一个我们此前利用微博数据和开源的Bert做预训练时随机抽出的一个中文训练实例，从中可以体会下上面讲的masked语言模型和下句预测任务。训练数据就长这种样子。



![img](E:\myBlog\source\_posts\E1vcQhzTsgbLqkF.jpg)

顺带讲解下Bert的输入部分，也算是有些特色。它的输入部分是个线性序列，两个句子通过分隔符分割，最前面和最后增加两个标识符号。每个单词有三个embedding:位置信息embedding，这是因为NLP中单词顺序是很重要的特征，需要在这里对位置信息进行编码；单词embedding,这个就是我们之前一直提到的单词embedding；第三个是句子embedding，因为前面提到训练数据都是由两个句子构成的，那么每个句子有个句子整体的embedding项对应给每个单词。把单词对应的三个embedding叠加，就形成了Bert的输入。



![img](E:\myBlog\source\_posts\wHfCcyhaiXPMx1d.jpg)

至于Bert在预训练的输出部分如何组织，可以参考上图的注释。



![img](E:\myBlog\source\_posts\B5wItXbYpPr619Z.jpg)

我们说过Bert效果特别好，那么到底是什么因素起作用呢？如上图所示，对比试验可以证明，跟GPT相比，双向语言模型起到了最主要的作用，对于那些需要看到下文的任务来说尤其如此。而预测下个句子来说对整体性能来说影响不算太大，跟具体任务关联度比较高。



![img](E:\myBlog\source\_posts\u9mKAEGjp4fa7ys.jpg)

最后，我讲讲我对Bert的评价和看法，我觉得Bert是NLP里里程碑式的工作，对于后面NLP的研究和工业应用会产生长久的影响，这点毫无疑问。但是从上文介绍也可以看出，从模型或者方法角度看，Bert借鉴了ELMO，GPT及CBOW，主要提出了Masked 语言模型及Next Sentence Prediction，但是这里Next Sentence Prediction基本不影响大局，而Masked LM明显借鉴了CBOW的思想。所以说Bert的模型没什么大的创新，更像最近几年NLP重要进展的集大成者，这点如果你看懂了上文估计也没有太大异议，如果你有大的异议，杠精这个大帽子我随时准备戴给你。如果归纳一下这些进展就是：首先是两阶段模型，第一阶段双向语言模型预训练，这里注意要用双向而不是单向，第二阶段采用具体任务Fine-tuning或者做特征集成；第二是特征抽取要用Transformer作为特征提取器而不是RNN或者CNN；第三，双向语言模型可以采取CBOW的方法去做（当然我觉得这个是个细节问题，不算太关键，前两个因素比较关键）。Bert最大的亮点在于效果好及普适性强，几乎所有NLP任务都可以套用Bert这种两阶段解决思路，而且效果应该会有明显提升。可以预见的是，未来一段时间在NLP应用领域，Transformer将占据主导地位，而且这种两阶段预训练方法也会主导各种应用。

另外，我们应该弄清楚预训练这个过程本质上是在做什么事情，本质上预训练是通过设计好一个网络结构来做语言模型任务，然后把大量甚至是无穷尽的无标注的自然语言文本利用起来，预训练任务把大量语言学知识抽取出来编码到网络结构中，当手头任务带有标注信息的数据有限时，这些先验的语言学特征当然会对手头任务有极大的特征补充作用，因为当数据有限的时候，很多语言学现象是覆盖不到的，泛化能力就弱，集成尽量通用的语言学知识自然会加强模型的泛化能力。如何引入先验的语言学知识其实一直是NLP尤其是深度学习场景下的NLP的主要目标之一，不过一直没有太好的解决办法，而ELMO/GPT/Bert的这种两阶段模式看起来无疑是解决这个问题自然又简洁的方法，这也是这些方法的主要价值所在。

对于当前NLP的发展方向，我个人觉得有两点非常重要，一个是需要更强的特征抽取器，目前看Transformer会逐渐担当大任，但是肯定还是不够强的，需要发展更强的特征抽取器；第二个就是如何优雅地引入大量无监督数据中包含的语言学知识，注意我这里强调地是优雅，而不是引入，此前相当多的工作试图做各种语言学知识的嫁接或者引入，但是很多方法看着让人牙疼，就是我说的不优雅。目前看预训练这种两阶段方法还是很有效的，也非常简洁，当然后面肯定还会有更好的模型出现。

完了，这就是自然语言模型预训练的发展史。







