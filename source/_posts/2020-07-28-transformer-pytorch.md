---
title: 2020-07-28-transformer解读-pytorch版本
date: 2020-07-28 16:43:03
tags: transformer
categories: transformer
top: 100
description: transformer-pytorch
---

### 前言

最近几天都在阅读哈佛pytorch实现transformer的代码，代码风格很好，很值得参考和研读。和实验室师兄又在一起讨论了几次，代码思路和实现过程基本都了解了，对于原论文 [“Attention is All You Need”](https://arxiv.org/abs/1706.03762) 中关于transformer模型的理解又深入了许多。果然要想了解模型，还是要好好研读实现代码。以便于后面自己结合模型的研究。s

本篇是对实现代码的注释，加上了自己的理解，也会有一些函数的介绍扩充。



#### 参考链接

> 解读的是哈佛的一篇transformer的pytorch版本实现
>
> http://nlp.seas.harvard.edu/2018/04/03/attention.html
>
> 参考另一篇博客
>
> http://fancyerii.github.io/2019/03/09/transformer-codes/
>
> Transformer注解及PyTorch实现（上）
>
> https://www.jiqizhixin.com/articles/2018-11-06-10
>
> Transformer注解及PyTorch实现（下）
>
> https://www.jiqizhixin.com/articles/2018-11-06-18
>
> 训练过程中的 Mask实现
>
> https://www.cnblogs.com/wevolf/p/12484972.html
>
> transformer综述
>
> [https://libertydream.github.io/2020/05/03/Transformer-%E7%BB%BC%E8%BF%B0/](https://libertydream.github.io/2020/05/03/Transformer-综述/)





### The Annotated Transformer



![这是一张图片](https://i.loli.net/2020/07/28/NUAyXWJ5DzHmjuv.png)





```python
# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

```





Transformer使用了Self-Attention机制，它在编码每一词的时候都能够注意(attend to)整个句子，从而可以解决长距离依赖的问题，同时计算Self-Attention可以用矩阵乘法一次计算所有的时刻，因此可以充分利用计算资源(CPU/GPU上的矩阵运算都是充分优化和高度并行的)。

### 模型结构

Most competitive neural sequence transduction models have an encoder-decoder structure [(cite)](https://arxiv.org/abs/1409.0473). Here, `the encoder maps an input sequence of symbol representations (x1,…,xn)(x1,…,xn) to a sequence of continuous representations z=(z1,…,zn)z=(z1,…,zn). Given z, the decoder then generates an output sequence (y1,…,ym)(y1,…,ym) of symbols one element at a time.` At each step the model is auto-regressive [(cite)](https://arxiv.org/abs/1308.0850), consuming the previously generated symbols as additional input when generating the next.



**EncoderDecoder定义了一种通用的Encoder-Decoder架构**，具体的Encoder、Decoder、src_embed、target_embed和generator都是构造函数传入的参数。这样我们**做实验更换不同的组件就会更加方便**。

```python
class EncoderDecoder(nn.Module): #定义的是整个模型 ，不包括generator
    
    """
   标准的Encoder-Decoder架构。这是很多模型的基础
    
    """
    """
    class里， init函数是实例化一个对象的时候用于初始化对象用的
    forward函数是在执行调用对象的时候使用， 需要传入正确的参数 
    在执行时候调用__call__方法，然后再call里再调用forward
    """
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        # encoder和decoder都是构造的时候传入的，这样会非常灵活
        self.encoder = encoder
        self.decoder = decoder
        # 源语言和目标语言的embedding，包括embedding层和position encode层
        self.src_embed = src_embed #源数据集的嵌入
        self.tgt_embed = tgt_embed #目标数据集的嵌入，作为decoder的输入
        """
        generator后面会讲到，就是根据Decoder的隐状态输出当前时刻的词
	    基本的实现就是隐状态输入一个全连接层，全连接层的输出大小是词的个数
		然后接一个softmax变成概率
        """
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        #首先调用encode方法对输入进行编码，然后调用decode方法解码
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        # 调用encoder来进行编码，传入的参数embedding的src和src_mask
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask) #目标是输入的一部分

    
class Generator(nn.Module):  #decoder后面的linear+softmax
    # 根据Decoder的隐状态输出一个词
	# d_model是Decoder输出的大小，vocab是词典大小 （数据语料有多少词 ）
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab) #全连接，作为softmax的输入。

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1) #softmax的log值
```



注：`Generator返回的是softmax的log值`。在PyTorch里为了计算交叉熵损失，有两种方法。第一种方法是使用**nn.CrossEntropyLoss()**，一种是使用**NLLLoss()**。很多开源代码里第二种更常见，

我们先看CrossEntropyLoss，它就是计算交叉熵损失函数，比如：

```python
criterion = nn.CrossEntropyLoss()

x = torch.randn(1, 5)
y = torch.empty(1, dtype=torch.long).random_(5)

loss = criterion(x, y)
```

比如上面的代码，假设是5分类问题，x表示模型的输出logits(batch=1)，而y是真实分类的下标(0-4)。实际的计算过程为：<img src="https://i.loli.net/2020/08/06/KyPspa4Cqef6m8Q.png" alt="image-20200806000621448" style="zoom: 67%;" />

比如logits是[0,1,2,3,4]，真实分类是3，那么上式就是：

<img src="https://i.loli.net/2020/08/06/i7mfUWAeHE5P1zd.png" alt="image-20200806000641945" style="zoom:67%;" />

因此我们也可以使用NLLLoss()配合F.log_softmax函数(或者nn.LogSoftmax，这不是一个函数而是一个Module了)来实现一样的效果：

```python
m = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss()
x = torch.randn(1, 5)
y = torch.empty(1, dtype=torch.long).random_(5)
loss = criterion(m(x), y)
```

NLLLoss(Negative Log Likelihood Loss)是计算负log似然损失。它输入的x是log_softmax之后的结果(长度为5的数组)，y是真实分类(0-4)，输出就是x[y]。因此上面的代码为：

```python
criterion(m(x), y)=m(x)[y]
```



The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.



![png](https://i.loli.net/2020/07/28/P3fSgRhrmFtlpxY.png)

### Encoder and Decoder Stacks

#### Encoder

Encoder和Decoder都是由N个相同结构的Layer堆积(stack)而成。**因此我们首先定义clones函数，用于克隆相同的SubLayer。**

这里使用了**nn.ModuleList**，ModuleList就像一个普通的Python的List，我们可以使用下标来访问它，它的好处是传入的ModuleList的所有Module都会注册的PyTorch里，这样Optimizer就能找到这里面的参数，从而能够用梯度下降更新这些参数。但是nn.ModuleList并不是Module(的子类)，因此它没有forward等方法，我们通常把它放到某个Module里。

```python
def clones(module, N):  #克隆N层，是个层数的列表。 copy.deepcopy是深复制， 一个改变不会影响另一个

	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) #复制N=6层
```



```python
class Encoder(nn.Module):  #定义编码器 
    
    #Encoder是N个EncoderLayer的stack
    def __init__(self, layer, N): # 根据make_model定义，layer = encoderlayer （sublayer）
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) #编码器有6层编码层，根据上述函数的定义，module=layer
        self.norm = LayerNorm(layer.size) #调用下面的LayerNorm。 分开定义是因为 LayerNorm = 2* layer
        
    def forward(self, x, mask): 
      	 #逐层进行处理
        for layer in self.layers: # x 在每一层中传递
            x = layer(x, mask)
        return self.norm(x) #最终encoder的返回值
```



```python
class LayerNorm(nn.Module): #add & norm部分  作为每一个子层的输出
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6): #feature = layer.size layer的形状
        super(LayerNorm, self).__init__()
        
        self.a_2 = nn.Parameter(torch.ones(features))  #将后面的tensor转换为可优化的参数
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps #很小的值

    def forward(self, x): # 平均值和标准差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 #输出
```

**不管是Self-Attention还是全连接层，都首先是LayerNorm，然后是Self-Attention/Dense，然后是Dropout，最好是残差连接。这里面有很多可以重用的代码，我们把它封装成SublayerConnection。**

------

That is, `the output of each sub-layer is LayerNorm(x+Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.` We apply dropout [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the output of each sub-layer, before it is added to the sub-layer input and normalized.

To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension `dmodel=512`.

```python
class SublayerConnection(nn.Module): #每一个编码层中的两个子层之间的连接
    """
	LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
	为了简单，把LayerNorm放到了前面，这和原始论文稍有不同，原始论文LayerNorm在最后。
	"""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
       #sublayer是传入的参数，参考DecoderLayer，它可以当成函数调用，这个函数的有一个输入参数
        return x + self.dropout(sublayer(self.norm(x))) #调用layernorm ，正则化之后再相加
```

这个类会构造LayerNorm和Dropout，但是Self-Attention或者Dense并不在这里构造，还是放在了EncoderLayer里，在forward的时候由EncoderLayer传入。这样的好处是更加通用，比如Decoder也是类似的需要在Self-Attention、Attention或者Dense前面后加上LayerNorm和Dropout以及残差连接，我们就可以复用代码。但是这里要求传入的sublayer可以使用一个参数来调用的函数(或者有__call__)。



------



forward调用sublayer[0] (这是SublayerConnection对象)的__call__方法，最终会调到它的forward方法，而这个方法需要两个参数，**一个是输入Tensor，一个是一个callable，并且这个callable可以用一个参数来调用**。而**self_attn函数需要4个参数(Query的输入,Key的输入,Value的输入和Mask)**，因此这里我们使用lambda的技巧把它变成一个参数x的函数(mask可以看成已知的数)。

  Callable 类型是可以被执行调用操作的类型。包含自定义函数等。自定义的函数比如使用def、lambda所定义的函数

```python

class EncoderLayer(nn.Module): #每一个编码层
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #每一层有2子层
        self.size = size

    def forward(self, x, mask):
      #attention层，括号里面是参数。接收来自attention的输出
    """
     lambda : atten()SublayerConnection里是作为sublayer出现的，而它的参数是norm(x),norm(x)的输出是一个向量x，
   所以atten的参数是只有一个x， 而在muitihead里面，k、q、v在函数里是要被重新根据x计算的
    """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 
        return self.sublayer[1](x, self.feed_forward) #x是atten+norm之后的输出，再ff输出
    """
    可以理解为
    z = lambda y: self.self_attn(y, y, y, mask)
	x = self.sublayer[0](x, z)
    """
```



#### Decoder

The decoder is also composed of a stack of `N=6` identical layers.

```python
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
     #memory: 编码器的输出 x是输入
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```



```python
class DecoderLayer(nn.Module): #每一层解码层
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3) #每一层有3个子层
        
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) #第一子层
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) #第二子层 
        return self.sublayer[2](x, self.feed_forward) #第三子层 
```

**src-attn和self-attn的实现是一样的，只不过使用的Query，Key和Value的输入不同。**普通的Attention(src-attn)的Query是下层输入进来的(来自self-attn的输出)，Key和Value是Encoder最后一层的输出memory；而Self-Attention的Query，Key和Value都是来自下层输入进来的。





------

Decoder和Encoder有一个关键的不同：Decoder在解码第t个时刻的时候只能使用**1…t时刻**的输入，而不能使用t+1时刻及其之后的输入。因此我们需要一个函数来产生一个Mask矩阵，所以代码如下：

注意： t时刻包括t时刻的输入

```python
def subsequent_mask(size):  #将i后面的mask掉
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') #triu 上三角
    return torch.from_numpy(subsequent_mask) == 0 #将numpy格式转换为tensor格式，判断是否为0， 输出布尔值
```





![png](https://i.loli.net/2020/08/08/7brnPfDJxsLBtvh.png)

它的输出：

```
print(subsequent_mask(5))
# 输出
  1  0  0  0  0
  1  1  0  0  0
  1  1  1  0  0
  1  1  1  1  0
  1  1  1  1  1
```

我们发现它输出的是一个方阵，对角线和下面都是1。**第一行只有第一列是1，它的意思是时刻1只能attend to输入1**，第三行说明时刻3可以attend to {1,2,3}而不能attend to{4,5}的输入，因为在真正Decoder的时候这是属于Future的信息。代码首先使用triu产生一个上三角阵：

```
0 1 1 1 1
0 0 1 1 1
0 0 0 1 1
0 0 0 0 1
0 0 0 0 0
```

然后需要把0变成1，把1变成0，这可以使用 matrix == 0来实现。

因为：布尔值True被索引求值为1，而False就等于0。



#### Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

We call our particular attention “`Scaled Dot-Product Attention`”. The input consists of queries and keys of dimension `dk`, and values of dimension `dv`. We compute the dot products of the query with all keys, divide each by `√dk`, and apply a softmax function to obtain the weights on the values.



![image-20200806015122441](https://i.loli.net/2020/08/06/O3UNSGF7Poa1w4Q.png)

**Attention可以看成一个函数，它的输入是Query,Key,Value和Mask，输出是一个Tensor**。其中输出是Value的加权平均，而权重来自Query和Key的计算。具体的计算如下图所示，计算公式为：

<img src="https://i.loli.net/2020/07/28/WaSfHnNdt2L1AXU.png" alt="image-20200728212241453" style="zoom:50%;" />

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) # query.size的最后一维
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:# 如果有mask
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None: #对p_attn进行dropout
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

我们知道, 在训练的时候, 我们是以 batch_size 为单位的, 那么就会有 padding, 一般我们取 pad == 0, 那么就会造成在 Attention 的时候, query 的值为 0, query 的值为 0, 所以我们计算的对应的 scores 的值也是 0, 那么就会导致 softmax 很可能分配给该单词一个相对不是很小的比例, 因此, 我们将 pad 对应的 score 取值为**负无穷**（普通的计算，score可以为负数？）, 以此来减小 pad 的影响. 



很容易想到, 在 decoder, **未预测的单词**也是用 padding 的方式加入到 batch 的, 所以使用的mask 机制与 padding 时mask 的机制是相同的, 本质上都是query 的值为0, 只是 mask 矩阵不同, 我们可以根据 decoder 部分的代码发现这一点.



------



我们使用一个**实际的例子跟踪一些不同Tensor的shape**，然后对照公式就很容易理解。比如**Q是(30,8,33,64)，其中30是batch，8是head个数，33是序列长度，64是每个时刻的特征数（size）。K和Q的shape必须相同的，而V可以不同，但是这里的实现shape也是相同的。**

```python
	scores = torch.matmul(query, key.transpose(-2, -1)) \
	/ math.sqrt(d_k)
```

上面的代码实现<img src="https://i.loli.net/2020/08/06/rLCJ7VFBAsmQb4x.png" alt="image-20200806014945713" style="zoom:50%;" />，和公式里稍微不同的是，这里的Q和K都是4d的Tensor，包括batch和head维度。**matmul会把query和key的最后两维进行矩阵乘法**，这样效率更高，如果我们要用标准的矩阵(二维Tensor)乘法来实现，那么需要遍历batch维和head维：

```python
	batch_num = query.size(0) # query.size(0)返回的是0维的数
	head_num = query.size(1)
	for i in range(batch_num):
		for j in range(head_num):
			scores[i,j] = torch.matmul(query[i,j], key[i,j].transpose())
```

而上面的写法一次完成所有这些循环，效率更高。**输出的score是(30, 8, 33, 33)**，前面两维不看，那**么是一个(33, 33)的attention矩阵a，aij表示时刻 i关注 j 的得分**(还没有经过softmax变成概率)。

**在编码器的attention中src_mask的作用！！！**

接下来是`scores.masked_fill(mask == 0, -1e9)`，用于**把mask是0的变成一个很小的数**，这样后面经过softmax之后的概率就很接近零(但是理论上还是用来很少一点点未来的信息)。

> masked_fill_(mask, value)：掩码操作
> masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），**元素是布尔值，value是要填充的值**，填充规则是mask中取值为True位置对应于self的相应位置用value填充。
>
> 注：参数mask必须与score的size相同或者两者是可广播(broadcasting-semantics)的
>
> 
>
> pytorch masked_fill方法简单理解 
>
>  https://blog.csdn.net/jianyingyao7658/article/details/103382654
>
> pytorch 广播语义(Broadcasting semantics) 
>
>  https://blog.csdn.net/qq_35012749/article/details/88308657



这里**mask是(30, 1, 1, 33)的tensor**，因为8个head的mask都是一样的，所有第二维是1，masked_fill时使用broadcasting就可以了。这里是self-attention的mask，所以每个时刻都可以attend到所有其它时刻，所有第三维也是1，也使用broadcasting。如果是普通的mask，那么mask的shape是(30, 1, 33, 33)。

这样讲有点抽象，我们可以举一个例子，为了简单，我们假设batch=2, head=8。第一个序列长度为3，第二个为4，那么self-attention的mask为(2, 1, 1, 4)，我们可以用两个向量表示：

```
1 1 1 0
1 1 1 1
```

它的意思是在self-attention里，第一个序列的任一时刻可以attend to 前3个时刻(因为第4个时刻是padding的)；而第二个序列的可以attend to所有时刻的输入。而Decoder的src-attention的mask为(2, 1, 4, 4)，我们需要用2个矩阵表示：(一个序列对应一个一维src_mask（1×4），  一个序列对应一个二维的tgt_mask（4×4）)

```
第一个序列的mask矩阵
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 0

第二个序列的mask矩阵
1 0 0 0
1 1 0 0 
1 1 1 0
1 1 1 1
```



接下来对score求softmax，把得分变成概率p_attn，如果有dropout还对p_attn进行Dropout(这也是原始论文没有的)。最后把p_attn和value相乘。p_attn是(30, 8, 33, 33)，value是(30, 8, 33, 64)，我们**只看后两维，(33x33) x (33x64)最终得到33x64。**



------



接下来就是输入怎么变成Q,K和V了，**对于每一个Head，都使用三个矩阵WQ,WK,WV把输入转换成Q，K和V。**然后**分别用每一个Head进行Self-Attention的计算，最后把N个Head的输出拼接起来，最后用一个矩阵WO把输出压缩一下。**具体计算过程为：

<img src="https://i.loli.net/2020/08/06/1IbPcFJeK8tsHqN.png" alt="image-20200806023820900" style="zoom: 67%;" />



详细结构如下图所示，输入Q，K和V经过多个线性变换后得到N(8)组Query，Key和Value，然后使用Self-Attention计算得到N个向量，然后拼接起来，**最后使用一个线性变换进行降维。**



<img src="https://i.loli.net/2020/08/08/a2gozSYGn8NOkpH.png" alt="png" style="zoom:67%;" />

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 不能整除就报错
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # # 所有h个head的mask都是相同的 
            mask = mask.unsqueeze(1) #在维度为1的位置添加一个维度，数字为1
        nbatches = query.size(0) #就是有多少batch的值
        
        # 1) 首先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
           #.view()表示重构张量的维度
         #注：因为每个Linear学习到的参数是不一样的。所以qkv三个也是不一样的
            
            
        # 2)使用attention函数计算 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) 把8个head的64维向量拼接成一个512的向量。然后再使用一个线性变换(512,521)，shape不变。 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```



我们先看构造函数，这里**d_model(512)是Multi-Head的输出大小**，因为有h(8)个head，因此每个head的d_k=512/8=64。接着我们构造4个(d_model ， d_model)的矩阵，后面我们会看到它的用处。最后是构造一个Dropout层。

然后我们来看forward方法。**输入的mask是(batch, 1, time)的，因为每个head的mask都是一样的，所以先用unsqueeze(1)变成(batch, 1, 1, time)**，mask我们前面已经详细分析过了。

接下来是**根据输入query，key和value计算变换后的Multi-Head的query，key和value**。这是通过下面的语句来实现的：

```python
query, key, value = \
		[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)	
			for l, x in zip(self.linears, (query, key, value))] # l(x): 调用nn.Linear函数
```

**zip(self.linears, (query, key, value))是把(self.linears[0],self.linears[1],self.linears[2])和(query, key, value)放到一起然后遍历。我们只看一个self.linears[0] (query)。根据构造函数的定义，self.linears[0]是一个(512, 512)的矩阵，而query是(batch, time, 512)，相乘之后得到的新query还是512(d_model)维的向量，然后用view把它变成(batch, time, 8, 64)。然后transponse成(batch, 8,time,64)，这是attention函数要求的shape。分别对应8个Head，每个Head的Query都是64维。**

> 1.一般来说，矩阵相乘，[a,b] x [b,c] = [a,c]
>
> 所以不同维度要进行处理，必须降维。例如 A 矩阵 [a,b,c], B 矩阵是[c,d]
>
> 这个时候就需要将 A 矩阵看成是 [axb, c] 与 [c,d] 进行相乘，得到结果。
>
> 2. Linear函数l(x)，应该就是 (batch*time,512)**(512,512)

Key和Value的运算完全相同，因此我们也分别得到8个Head的64维的Key和64维的Value。接下来**调用attention函数，得到x和self.attn。其中x的shape是(batch, 8, time, 64)，而attn是(batch, 8, time, time)。**

**x.transpose(1, 2)把x变成(batch, time, 8, 64)，然后把它view成(batch, time, 512)，其实就是把最后8个64维的向量拼接成512的向量。最后使用self.linears[-1]对x进行线性变换，self.linears[-1]是(512, 512)的，因此最终的输出还是(batch, time, 512)。我们最初构造了4个(512, 512)的矩阵，前3个用于对query，key和value进行变换，而最后一个对8个head拼接后的向量再做一次变换。**



#### A0ttention在模型中的应用

在Transformer里，有3个地方用到了MultiHeadedAttention：

- Encoder的Self-Attention层

  **query，key和value都是相同的值**，来自下层的输入。Mask都是1(当然padding的不算)。

- Decoder的Self-Attention层

  **query，key和value都是相同的值**，来自下层的输入。但是Mask使得它不能访问未来的输入。

- Encoder-Decoder的普通Attention

  **query来自下层的输入，而key和value相同**，是Encoder最后一层的输出，而Mask都是1。

  

### Position-wise 前馈网络

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. `This consists of two linear transformations with a ReLU activation in between.`

全连接层有两个线性变换以及它们之间的ReLU激活组成：

<img src="https://i.loli.net/2020/08/08/PU96rciRsWxOCKp.png" alt="image-20200728231445307" style="zoom:50%;" />

全连接层的输入和输出都是d_model(512)维的，中间隐单元的个数是d_ff(2048)维

```python
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

### Embeddings 和 Softmax



**输入的词序列都是ID序列，我们需要Embedding**。源语言和目标语言都需要Embedding，此外我们需要一个线性变换把隐变量变成输出概率，这可以通过前面的类Generator来实现。我们这里实现Embedding：

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) #将字典vocab大小映射到d_model大小
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

注意的就是forward处理使用nn.Embedding对输入x进行Embedding之外，还除以了sqrt(d_model) （开方）



### 位置编码

位置编码的公式为：

 <img src="https://i.loli.net/2020/08/08/WUpXhHsK3S1jCqn.png" alt="image-20200728232133981" style="zoom:50%;" />

<img src="https://i.loli.net/2020/08/08/XOZPy89KVi1xjTh.png" alt="image-20200728232255029" style="zoom:50%;" />

 where `pos` is the position and `i` is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. 

假设输入是ID序列长度为10，**如果输入Embedding之后是(10, 512)，那么位置编码的输出也是(10, 512)。**上式中pos就是位置(0-9)，512维的偶数维使用sin函数，而奇数维使用cos函数。这种位置编码的好处是：PE_pos+k可以表示成 PE_pos的线性函数，这样网络就能容易的学到相对位置的关系。

```python
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
```

图是一个示例，向量的大小d_model=20，我们这里画出来第4、5、6和7维(下标从零开始)维的图像，最大的位置是100。我们可以看到它们都是正弦(余弦)函数，而且周期越来越长。



![png](https://i.loli.net/2020/08/08/TfDHKvnM3emYysL.png)



前面我们提到位置编码的好处是PE_pos+k可以表示成 P_Epos的线性函数，我们下面简单的验证一下。我们以第i维为例，为了简单，我们把<img src="https://i.loli.net/2020/08/06/iEoDOvKzB42N6Xe.png" alt="image-20200806104700979" style="zoom: 67%;" />记作Wi，这是一个常量。



<img src="https://i.loli.net/2020/08/06/E9h2vXIDK1MAjUg.png" alt="image-20200806104725624" style="zoom:67%;" />

我们发现PE_pos+k 确实可以表示成 PE_pos的线性函数。



In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of `Pdrop=0.1`.

```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        #之所以用log再exp,可能是考虑到数值过大溢出的问题
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
```

代码可以参考公式，调用了`Module.register_buffer函数`。这个函数的作用是创建一个buffer，比如这里把pe保存下来。register_buffer通常用于保存一些模型参数之外的值，比如在BatchNorm中，我们需要保存running_mean(Moving Average)，它不是模型的参数(不用梯度下降)，但是模型会修改它，而且在预测的时候也要使用它。这里也是类似的，pe是一个提前计算好的常量，我们在forward要用到它。我们在构造函数里并没有把pe保存到self里，但是在forward的时候我们却可以直接使用它(self.pe)。如果我们保存(序列化)模型到磁盘的话，PyTorch框架也会帮我们保存buffer里的数据到磁盘，这样反序列化的时候能恢复它们



### 完整模型

> Here we `define a function that takes in hyperparameters and produces a full model.`

```python


def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1): #d_ff： feedforward的维度
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg. 随机初始化参数，这非常重要
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# 示例: 对model简单输入参数
tmp_model = make_model(10, 10, 2)

```

首先把copy.deepcopy命名为c，这样使下面的代码简洁一点。然后构造MultiHeadedAttention，PositionwiseFeedForward和PositionalEncoding对象。接着就是构造EncoderDecoder对象。它需要5个参数：Encoder、Decoder、src-embed、tgt-embed和Generator。

我们先看后面三个简单的参数，Generator直接构造就行了，它的作用是把模型的隐单元变成输出词的概率。而src-embed是一个Embeddings层和一个位置编码层c(position)，tgt-embed也是类似的。

最后我们来看Decoder(Encoder和Decoder类似的)。Decoder由N个DecoderLayer组成，而DecoderLayer需要传入self-attn, src-attn，全连接层和Dropout。因为所有的MultiHeadedAttention都是一样的，因此我们直接deepcopy就行；同理所有的PositionwiseFeedForward也是一样的网络结果，我们可以deepcopy而不要再构造一个。



### 训练

This section describes the training regime for our models.

> We stop for a quick interlude to introduce some of the tools needed to train a standard encoder decoder model. First `we define a batch object that holds the src and target sentences for training, as well as constructing the masks.`

#### Batches 和 Masking

`mask 矩阵来自 batch`

`self.src_mask = (src != pad).unsqueeze(-2)` 也就是说, 源语言的 **mask 矩阵的维度是 (batch_size, 1, length)**, 那么为什么 `attn_shape = (batch_size, size, size)` 呢? 可以这么解释, **在 encoder 阶段的 Self_Attention 阶段, 所有的 Attention 是可以同时进行的, 把所有的 Attention_result 算出来, 然后用同一个 mask vector * Attention_result 就可以了**, 但是在 decoder 阶段却不能这么做, 我们需要关注的问题是:

> 根据已经预测出来的单词预测下面的单词, 这一过程**是序列的**,
>
> 而我们的计算是**并行**的, 所以这一过程中, 必须要引入矩阵. 也就是上面的 subsequent_mask() 函数获得的矩阵.

这个矩阵也很形象, 分别表示已经预测的单词的个数为, 1, 2, 3, 4, 5.

然后我们将以上过程反过来过一篇, 就很明显了, 在 batch阶段获得 mask 矩阵, 然后和 batch 一起训练, 在 encoder 与 deocder 阶段实现 mask 机制.



> - mask在Batch中定义，src_mask.size (30,1,10) ,  trg_mask.size(30,10,10)
>
> - 然后在MultiHeadedAttention中`mask = mask.unsqueeze(1)`又扩维了，
>
>   其中src_mask.size(30,1,1,10) ,trg_mask.size(30,1,10,10)
>
> - src_mask.size满足attention中的维度，所以可以对score进行mask
>
>    src_mask还在解码器的第1子层用到，相同的原理
>
> - trg_mask在解码器的第0子层用到，满足要求

```python
class Batch: #定义每一个batch中的src、tgt、mask
    #trg = tgt: 真实的标签序列  out ： 预测的单词  
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0): 
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) #扩充维度 倒数第二维增加值为1 size=(30,1,10)
        #并且非零值全部赋值为1
        
         # 在预测的时候是没有 tgt 的,此时为 None 此时trg是tgt的形参
        if trg is not None:
            self.trg = trg[:, :-1] #trg.size(30,9) ，在预测中，会提前输入起始符到ys中
            """
              trg.size(30,9) 这里去掉的最后一个单词, 不是真正的单词, 而是标志 '<eos>' , 						输入与输出都还有一个 '<sos>' 在句子的开头,  是decoder的输入，
            需要进行mask，使得Self-Attention不能访问未来的输入。最后一个词不需要用到trg
            """
    	    self.trg_y = trg[:, 1:] # trg_y.size(30,9) 
            #trg_y: 最后的结果。用于loss中的比较。 去掉开头的'<sos>'，是decoder的输出
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum() #不为0的总数 30*9 = 270
    
    @staticmethod
    def make_std_mask(tgt, pad): #tgt_mask.size(30,9,9)，每一个序列都是一个9*9的矩阵
        "Create a mask to hide padding and future words."
        #"创建Mask，使得我们不能attend to未来的词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
```

Batch构造函数的输入是src和trg，后者可以为None，因为再预测的时候是没有tgt的。

我们用一个例子来说明Batch的代码，这是训练阶段的一个Batch，**src是(48, 20)**，48是batch大小，而20是最长的句子长度，其它的不够长的都padding成20了。而**trg是(48, 25)**，表示翻译后的最长句子是25个词，不足的也padding过了。

我们首先看src_mask怎么得到，(src != pad)把src中大于0的时刻置为1，这样表示它可以attend to的范围。然后unsqueeze(-2)把src_mask变成(48/batch, 1, 20/time)。它的用法参考前面的attention函数。

对于训练来说(Teaching Forcing模式)，Decoder有一个输入和一个输出。**比如句子”<sos> it is a good day <eos>”，输入会变成”<sos> it is a good day”，而输出为”it is a good day <eos>”。对应到代码里，self.trg就是输入，而self.trg_y就是输出。**接着对输入self.trg进行mask，使得Self-Attention不能访问未来的输入。这是通过make_std_mask函数实现的，这个函数会调用我们之前详细介绍过的subsequent_mask函数。最终得到的**trg_mask的shape是(48/batch, 24, 24)**，表示24个时刻的Mask矩阵，这是一个对角线以及之下都是1的矩阵，前面已经介绍过了。

注意**src_mask的shape是(batch, 1, time)**，而**trg_mask是(batch, time, time)**。因为src_mask的每一个时刻都能attend to所有时刻(padding的除外)，一次只需要一个向量就行了，而trg_mask需要一个矩阵。



#### Training Loop

```python
def run_epoch(data_iter, model, loss_compute): #返回total_loss / total_tokens 。是一个数值，损失计算
    #遍历一个epoch的数据
    "Standard Training and Logging Function"
    start = time.time() #开始时间，计算用时
    total_tokens = 0 
    total_loss = 0 
    tokens = 0 
    for i, batch in enumerate(data_iter): #每一步data_iter（gen_data），实例化batch数据用于学习.进行20次
        #gen_data返回的是20个Batch，通过enumerate实例化20个batch 
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask) #调用EncoderDecoder的实例化model，解码器作为输出
        loss = loss_compute(out, batch.trg_y, batch.ntokens) #计算出一个batch中的loss。 trg_y是标准值。ntokens作为norm
        total_loss += loss #loss叠加。进行20次
        total_tokens += batch.ntokens 
        tokens += batch.ntokens
        if i % 50 == 1: #i从0开始的，当i=1的时候，进行了一次batch，所以这里计算的就是一次batch所用的时间。而要进行20次。  50是随机设置
            elapsed = time.time() - start #计算一共用时
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % 
                    (i, loss / batch.ntokens, tokens / elapsed)) #所有batch中的loss和ntoken,即一个epoch中
            start = time.time() # 重置时间
            tokens = 0
    return total_loss / total_tokens
```

它遍历一个epoch的数据，然后调用forward，接着用loss_compute函数计算梯度，更新参数并且返回loss。这里的loss_compute是一个函数，它的输入是模型的预测out，真实的标签序列batch.trg_y和batch的词个数。实际的实现是MultiGPULossCompute类，这是一个callable。本来计算损失和更新参数比较简单，但是这里为了实现多GPU的训练，这个类就比较复杂了。



#### Training Data 和 Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens. For English- French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary.

Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

> We will use torch text for batching. This is discussed in more detail below. Here we create batches in a torchtext function that ensures our batch size padded to the maximum batchsize does not surpass a threshold (25000 if we have 8 gpus).

```python
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
```

#### 硬件 和 训练进度

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models, step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

#### Optimizer

We used the `Adam optimizer` [(cite)](https://arxiv.org/abs/1412.6980) with β1=0.9β1=0.9, β2=0.98β2=0.98 and ϵ=10−9ϵ=10−9. We varied the learning rate over the course of training, according to the formula: lrate=d−0.5model⋅min(step_num−0.5,step_num⋅warmup_steps−1.5)lrate=dmodel−0.5⋅min(step_num−0.5,step_num⋅warmup_steps−1.5) This corresponds to increasing the learning rate linearly for the first warmupstepswarmupsteps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmupsteps=4000warmupsteps=4000.

> Note: This part is very important. Need to train with this setup of the model.

```python
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

> Example of the curves of this model for different model sizes and for optimization hyperparameters.

```python
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None), 
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None
```

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_69_0.png)

#### Regularization

##### Label Smoothing

During training, we employed label smoothing of value ϵls=0.1ϵls=0.1 [(cite)](https://arxiv.org/abs/1512.00567). This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

> We implement label smoothing using the KL div loss. Instead of using a one-hot target distribution, we create a distribution that has `confidence` of the correct word and the rest of the `smoothing` mass distributed throughout the vocabulary.

```python
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)  #KL散度
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
```

> Here we can see an example of how the mass is distributed to the words based on confidence.

```python
# Example of label smoothing.
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
         Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
None
```

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_74_0.png)

> Label smoothing actually starts to penalize the model if it gets very confident about a given choice.

```python
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    #print(predict)
    return crit(Variable(predict.log()),
                 Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
None
```

![png](http://nlp.seas.harvard.edu/images/the-annotated-transformer_76_0.png)

### **总结**

transformer模型主要分为两大部分, 分别是编码器和解码器, 编码器负责把自然语言序列映射成为隐藏层(下图中第2步用九宫格比喻的部分), 含有自然语言序列的数学表达. 然后解码器把隐藏层再映射为自然语言序列, 从而使我们可以解决各种问题, 如情感分类, 命名实体识别, 语义关系抽取, 摘要生成, 机 器翻译等等, 下面我们简单说一下下图的每一步都做了什么:

> 1.输入自然语言序列到编码器: Why do we work?(为什么要工作); 
>
> 2.编码器输出的隐藏层, 再输入到解码器; 
>
> 3.输入<𝑠𝑡𝑎𝑟𝑡><start>(起始)符号到解码器; 
>
> 4.得到第一个字"为"; 
>
> 5.将得到的第一个字"为"落下来再输入到解码器; 
>
> 6.得到第二个字"什"; 
>
> 7.将得到的第二字再落下来, 直到解码器输出<𝑒𝑛𝑑><end>(终止符), 即序列生成完成.



<img src="https://i.loli.net/2020/08/06/1pea3WSThisHBql.png" alt="image-20200806233205808" style="zoom:67%;" />







![transformer](https://i.loli.net/2020/08/07/ZGa1snNULJtjFWS.png)



原始data数据是：(30,10)

src: (30,10)  trg:(30,10) 

在encoder中，

embedding： 参数x就是 src （30,10） 经过处理之后， x:（30,10,512） -> 即输入给encoder的x：(30,10,512)

经过encoder各个层处理之后，输出的（30，10,512）  memory是encoder的输出，但是为什么memory：（1,10,512） ??? 因为在预测时 ，src是（1,10），不是（30,10）所以memory是（1,10,512）



decoder中：输入来自 memory 和 trg_emd

embedding ： 参数x是trg（30,9），经过处理之后，x：（30,9,512)   

经过decoder各个层处理之后，输出的（30，9 , 512）  



再经过generator层之后，x：（30,9,11） 



在预测的时候是（1，1,512），不是（1,9,512），在预测完generator之后，（1,11），选一个最大的。

因为是一个数字一个数字预测输出的，所以是1，不是9



 







### 第一个例子

> We can begin by trying out a simple copy-task. Given a random set of input symbols from a small vocabulary, the goal is to generate back those same symbols.

#### Synthetic Data

```python
def data_gen(V, batch, nbatches): # batch=30:一次输入多少， nbatch=20：输入多少次
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches): #一共循环nbatches个，在每一个是一个batch
		#from_numpy ： 将numpy数据转换为tensor
		#注：生成返回的tensor会和ndarry共享数据，任何对tensor的操作都会影响到ndarry
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))) #1是产生的最小值，V=11是最大值，size是形状（batch，10）。生成（batch，10）的矩阵，矩阵的每一个元素都是1~V-1之间  （取不到V）
        data[:, 0] = 1 #将第0列的值赋值为1
        # Variable 就是一个存放值， 里面的值会不停的变化.  存放的是Torch 的 Tensor . 如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.  
        #requires_grad： 是否参与误差反向传播, 要不要计算梯度
        src = Variable(data, requires_grad=False) #size(batch,10) 和data的值完全一样
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)#yield就是return一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始
        #batch返回的是trg_mask 
```

#### Loss Computation

```python
class SimpleLossCompute: #loss计算以及更新。调用LabelSmoothing，使用KL散度
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator #解码器后的生成函数
        self.criterion = criterion # LabelSmoothing（计算loss KLDivLoss KL散度）的实例化
        self.opt = opt # NoamOpt（优化）的实例化
        
    def __call__(self, x, y, norm):
        x = self.generator(x) #解码器的输出
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm  #计算loss
        loss.backward() #将loss反向传播。loss是标量，根据链式法则自动计算出叶子节点的梯度值
        if self.opt is not None: #存在优化
            self.opt.step() #调用opt的step函数。 adam优化，，更新参数
            self.opt.optimizer.zero_grad() #把梯度置零，也就是把loss关于weight的导数变成0.
        return loss.data[0] * norm
```

#### Greedy Decoding

```python
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0) #LabelSmoothing是KL散度实现的
model = make_model(V, V, N=2) #src_vocab=11, tgt_vocab=11，覆盖N=2
# 对模型参数进行更新优化，使用Adam优化
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

#model.eval()，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
#model.train() 让model变成训练模式，此时dropout和batch normalization的操作在训练起到防止网络过拟合的问题

for epoch in range(10): #一共10大份， model.train()打印1行，model.eval()打印1行
    model.train()
    #调用run_epoch(data_iter, model, loss_compute)函数
    #返回total_loss / total_tokens 。返回值可以没有接收，不会报错
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    #print接收run_epoch的返回值 在输出的第三行
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))
Epoch Step: 1 Loss: 3.023465 Tokens per Sec: 403.074173
Epoch Step: 1 Loss: 1.920030 Tokens per Sec: 641.689380
1.9274832487106324
Epoch Step: 1 Loss: 1.940011 Tokens per Sec: 432.003378
Epoch Step: 1 Loss: 1.699767 Tokens per Sec: 641.979665
1.657595729827881
Epoch Step: 1 Loss: 1.860276 Tokens per Sec: 433.320240
Epoch Step: 1 Loss: 1.546011 Tokens per Sec: 640.537198
1.4888023376464843
Epoch Step: 1 Loss: 1.682198 Tokens per Sec: 432.092305
Epoch Step: 1 Loss: 1.313169 Tokens per Sec: 639.441857
1.3485562801361084
Epoch Step: 1 Loss: 1.278768 Tokens per Sec: 433.568756
Epoch Step: 1 Loss: 1.062384 Tokens per Sec: 642.542067
0.9853351473808288
Epoch Step: 1 Loss: 1.269471 Tokens per Sec: 433.388727
Epoch Step: 1 Loss: 0.590709 Tokens per Sec: 642.862135
0.5686767101287842
Epoch Step: 1 Loss: 0.997076 Tokens per Sec: 433.009746
Epoch Step: 1 Loss: 0.343118 Tokens per Sec: 642.288427
0.34273059368133546
Epoch Step: 1 Loss: 0.459483 Tokens per Sec: 434.594030
Epoch Step: 1 Loss: 0.290385 Tokens per Sec: 642.519464
0.2612409472465515
Epoch Step: 1 Loss: 1.031042 Tokens per Sec: 434.557008
Epoch Step: 1 Loss: 0.437069 Tokens per Sec: 643.630322
0.4323212027549744
Epoch Step: 1 Loss: 0.617165 Tokens per Sec: 436.652626
Epoch Step: 1 Loss: 0.258793 Tokens per Sec: 644.372296
0.27331129014492034
```

> This code predicts a translation using greedy decoding for simplicity.

```python
#预测过程
#预测的时候没有用tgt（标准值），而是每次解码器的输入都是ys，是预测的值
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask) #memory是编码器的输出 。是一个矩阵
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data) #填充输出开始符，和src的类型一样。对预测的句子进行初始化 ys =1 （1,1）
    for i in range(max_len-1): #0~8 对每一个词都进行预测
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
         # ys 的维度是 batch_size * times （固定的）,   所以target_mask 矩阵必须是ys.size(1),所以是 times * times
        # 根据 decoder 的训练步骤, 这里的 out 输出就应该是 batch_size * (times+1) 的矩阵
        
        prob = model.generator(out[:, -1]) #generator返回的是softmax
          # out[:, -1] 这里是最新的一个单词的 embedding 向量
        # generator 就是产生最后的 vocabulary 的概率, 是一个全连接层
        
        _, next_word = torch.max(prob, dim = 1) # torch.max:按维度dim 返回最大值，并且会返回索引。next_data接收											#索引
        next_word = next_word.data[0]
        # 将句子拼接起来  .type_as: 将tensor强制转换为src.data 格式的
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
    1     2     3     4     5     6     7     8     9    10
[torch.LongTensor of size 1x10]
```

### 真实例子

> Now we consider a real-world example using the IWSLT German-English Translation task. This task is much smaller than the WMT task considered in the paper, but it illustrates the whole system. We also show how to use multi-gpu processing to make it really fast.

```python
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de
```

#### Data Loading

> We will load the dataset using torchtext and spacy for tokenization.
>
> 用torchtext来加载数据集 ， 用spacy来分词



<img src="https://i.loli.net/2020/08/07/teSG1hufEjF4Zkv.png" alt="image-20200807001353729" style="zoom: 67%;" />





torchtext组件流程：

> - 定义Field：声明如何处理数据，主要包含以下数据预处理的配置信息，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等等
> - 定义Dataset：用于得到数据集，继承自pytorch的Dataset。此时数据集里每一个样本是一个 经过 Field声明的预处理 预处理后的 wordlist
> - 建立vocab：在这一步建立词汇表，词向量(word embeddings)
> - 构造迭代器Iterator：: 主要是数据输出的模型的迭代器。构造迭代器，支持batch定制用来分批次训练模型。

```python
# For data loading.
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de') #加载德语语言模型
    spacy_en = spacy.load('en') #加载英语语言模型
	
    
    """
   在文本处理的过程中，spaCy首先对文本分词，原始文本在空格处分割，类似于text.split(' ')，然后分词器（Tokenizer）从左向右依次处理token
    """
    def tokenize_de(text): #Tokenizer:分词器  进行德语分词  
        #text：输入的段落句子  tok.text：分后的token词
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text): # 进行英语分词
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'  #开始符
    EOS_WORD = '</s>' #终止符
    BLANK_WORD = "<blank>" #空格
    
    # 构建Filed对象，声明如何处理数据。主要包含以下数据预处理的配置信息，比如指定分词方法，是否转成小写，		#起始字符，结束字符，补全字符以及词典等等
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD) #得到源句子
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,  
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100 #最大长度
    
    # https://s3.amazonaws.com/opennmt-models/iwslt.pt 数据集
    #同时对训练集和验证集还有测试集的构建，此时数据集里每一个样本是一个 经过 Field声明的预处理 预处理后的 	#wordlist
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en')   # 构建数据集所需的数据集
        , fields=(SRC, TGT),  #如何赋值给train那三个的？？？？
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)  #源句子和目标句子长度小于100的筛选出来
    
    MIN_FREQ = 2 #定义最小频率
    
    #建立词汇表，词向量(word embeddings)。即需要给每个单词编码，然后输入模型
    #bulid_vocab()方法中传入用于构建词表的数据集
    SRC.build_vocab(train.src, min_freq=MIN_FREQ) 
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    
    #一旦运行了这些代码行，SRC.vocab.stoi将是一个词典，其词汇表中的标记作为键，而其对应的索引作为值； 	#SRC.vocab.itos将是相同的字典，其中的键和值被交换。
```



> 批训练对于速度来说很重要。希望批次分割非常均匀并且填充最少。 要做到这一点，我们**必须修改torchtext默认的批处理函数**。 这部分代码修补其默认批处理函数，以确保我们搜索足够多的句子以构建紧密批处理。  一般来说直接调用`BucketIterator` （训练用）和 `Iterator`（测试用） 即可
>
> `BucketIterator`和`Iterator`的区别是，BucketIterator尽可能的把长度相似的句子放在一个batch里面。

#### Iterators

```python
"""
定义一个迭代器，该迭代器将相似长度的示例批处理在一起。 在为每个新纪元(epoch)生产新鲜改组的批次时，最大程度地减少所需的填充量。
"""
class MyIterator(data.Iterator):
    def create_batches(self):
        #在train的时候，要进行sort，尽量减少padding
        #目的是自动进行shuffle和padding，并且为了训练效率期间，尽量把句子长度相似的shuffle在一起。
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key), #按照词的数大小排序
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b #b就是batch， 类比上述的gen_data函数
            self.batches = pool(self.data(), self.random_shuffler) #调用pool
            
         #在valid+test(验证集和测试集)的时候  和上面具体区别在哪？？？？
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):  #pad_idx：空格键
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)#为什么要进行
    return Batch(src, trg, pad_idx) #调用上述的Batch类   pad_idx就是pad
```

#### Multi-GPU Training



> 最后为了真正地快速训练，将使用多个GPU。 这部分代码实现了多GPU字生成，它不是Transformer特有的。 其**思想是将训练时的单词生成分成块，以便在许多不同的GPU上并行处理。** 我们使用PyTorch并行原语来做到这一点：
>
> 
>
> - replicate -复制 - 将模块拆分到不同的GPU上
> - scatter -分散 - 将批次拆分到不同的GPU上
> - parallel_apply -并行应用 - 在不同GPU上将模块应用于批处理
> - gather - 聚集 - 将分散的数据聚集到一个GPU上
> - nn.DataParallel - 一个特殊的模块包装器，在评估之前调用它们。

```python
# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize
```

> Now we create our model, criterion, optimizer, data iterators, and paralelization

```python
# GPUs to use
devices = [0, 1, 2, 3]
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)
None
```

> Now we **train the model**. I will play with the warmup steps a bit, but everything else uses the default parameters. On an AWS p3.8xlarge with 4 Tesla V100s, this runs at ~27,000 tokens per second with a batch size of 12,000
>
> 在具有4个Tesla V100 GPU的AWS p3.8xlarge机器上，每秒运行约27,000个词，批训练大小为12,000。

#### Training the System

```python
#!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
#进行train和eval
if False: # false存在的意义在哪？？？ 使用GPU？
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model_par, 
                  MultiGPULossCompute(model.generator, criterion, 
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                          model_par, 
                          MultiGPULossCompute(model.generator, criterion, 
                          devices=devices, opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt") #加载所有的tensor到CPU	
```

> Once trained we can decode the model to produce a set of translations. Here we simply translate the first sentence in the validation set. This dataset is pretty small so the translations with greedy search are reasonably accurate.

```python
#类比于run_epoch函数  
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break
Translation:	<unk> <unk> . In my language , that means , thank you very much . 
Gold:	<unk> <unk> . It means in my language , thank you very much . 
```

