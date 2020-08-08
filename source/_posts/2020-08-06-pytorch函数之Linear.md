---
title: 2020-08-06-pytorch函数之nn.Linear
date: 2020-08-06 10:29:09
tags: pytorch
top:
categories: 
description: Linear函数
---

# 

class torch.nn.Linear（in_features，out_features，bias = True ）

对传入数据应用线性变换：y = A x+ b

 

参数：

in_features - 每个输入样本的大小

out_features - 每个输出样本的大小

bias - 如果设置为False，则图层不会学习附加偏差。默认值：True



代码：

```
m = nn.Linear(20, 30)

input = autograd.Variable(torch.randn(128, 20))

output = m(input)

print(output.size())
```



输出：

```
torch.Size([128, 30])
```

分析:

output.size()=矩阵size(128,20)*矩阵size（20,30）=(128,30)