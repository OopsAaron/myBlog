---
title: 2020-09-14-修改pytorch版transformer代码
mathjax: true
date: 2020-09-14 16:26:17
tags:
top:
categories:
description: 记录修改过程
---



~~源代码是`main3.py`  ,在此基础上进行修改，修改后文件为`main3-2.py`~~

740中`annotated-transformer`中`main.py`和哈佛的一样

复制到了本地`main.py`  ,再复制到`annotated-transformer1`中的main.py

**所以改前的代码是`main.py`  ，改后的代码是`main-1.py`**

注：

**`python main.py >main.txt 2>&1`，在将结果重定向到main.txt中，会覆盖main.txt之前的内容**

**每次跑实验的预测都是不一样的，但是都是和输入差不多**



1. 将`attention`函数去掉，合并到`MultiHeadedAttention`中，服务器上测试**可行**  





