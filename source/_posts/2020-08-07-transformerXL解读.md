---
title: 2020-08-07-transformerXL解读
date: 2020-08-07 18:20:34
tags:
top:
categories:
description: 最近读的论文transformer-XL
---



 Transformer最大的问题：在语言建模时的设置受到固定长度上下文的限制。

本文提出的Transformer-XL，使学习不再仅仅依赖于定长，且不破坏时间的相关性。

Transformer-XL包含segment-level 循环机制和positional编码框架。不仅可以捕捉长时依赖，还可以解决上下文断片问题 fragmentation problem。可以学到比RNNs长80%的依赖，比vanilla Transformers长450%。在长短序列上都取得了更好的结果。与vanilla Transformer相比，Transformer-XL的另一个优势是它可以被用于单词级和字符级的语言建模。