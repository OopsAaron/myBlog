---
title: 2020-08-12-更换hexo渲染器支持Latex
mathjax: true
date: 2020-08-12 23:44:37
tags: [hexo , Latex]
top: 80
categories: hexo 
description: 更换hexo渲染器支持Latex
---

### 前言

> `LATEX` 是一种基于 `TEX` 的排版系统，利用这种格式，可以迅速生成复杂表格和数学公式等，对于我们写博客帮助十分大。

### 初级

#### 版本

我使用的是hexo +  Next ，版本号如下：

```python
Hexo: 4.2.1  #在 ~\package.json中查看
NexT: 7.8.0  # 在~\themes\next\package.json中查看
```

#### MathJax 插件

渲染数学公式需要MathJax插件，有些 Hexo 主题自带 MathJax 插件，例如 [NexT](http://theme-next.iissnan.com/)只需启用该插件即可

如果没有的话，可以手动安装：

```
npm install hexo-math --save
```

#### 启用

NexT 主题的 MathJax 插件默认是禁用的，打开主题配置文件，将`mathjax`的`enable` 的值改为 `true` 即可启用 `MathJax`

注意 `per_page` 上面的注释，注释表明了，MathJax 只渲染在文件前端注明 `mathjax: true` 字段的文章，

所以为了以后在每一个新建的文件都默认带有`mathjax: true` ，可以在`~\scaffolds\post.md`中修改文章头部，添加`mathjax: true` 即可



#### 效果

行内公式：

```
这是一个行内公式：$sin^2\theta + cos^2\theta = 1$
```

效果：

这是一个行内公式：$sin^2\theta + cos^2\theta = 1$

整行公式：

```
$$sin^2\theta + cos^2\theta = 1$$
```

效果：

$$sin^2\theta + cos^2\theta = 1$$



#### 获取Latex公式：Mathpix Snipping Tool

如果要写Latex公式的话，需要掌握很多Latex语法，操作起来比较麻烦，学习成本也高。再加上平时我都是直接copy所读论文中的公式，于是我使用了`Mathpix Snipping Tool` 软件。

`Mathpix Snipping Tool` ： 通过对所要获取的公式进行截图，可以得到公式的Latex表达形式，复制到博客中即可。操作简单高效。使用方法不再赘述，网上资源很多。



### 高级

#### 危：渲染复杂LaTeX数学公式出现问题

发现一个问题就是编辑好的LaTex公式可以在 Markdown 编辑器（Typora）中显示出来，但部署之后，公式出现无法被渲染

之后通过Google之后，发现问题的一些源头

> 将`MathJax`改为true后发现，**只能渲染部分简单的公式**，对于稍微复杂一点的，特别是有下划线 ' _ ' 符号的公式，几乎都无法被渲染。
>
> hexo默认使用marked.js去解析我们写的markdown，比如一些符号，_代表斜体，会被处理为*标签，比如x_i在开始被渲染的时候，处理为xi，比如__init__会被处理成**init。***
>
> 
>
> Hexo 对 Markdown 文件的处理实际上分为两个步骤：
>
> 1. Hexo 中的 Markdown 引擎把 Markdown 变为 html 文件
> 2. MathJax 负责解释 html 的数学公式

所以现有的hexo渲染器是无法解决当前的问题，所以就要更换渲染器

#### 下载pandoc

打开powershell，输入以下命令行

```shell
pip install Pandoc
```



#### 安装 hexo-renderer-pandoc

在blog文件夹下打开git bash

```shell
npm uninstall hexo-renderer-marked --save #卸载旧版本
#因为之前为了支持emoji，我已经将hexo-renderer-marked换成了hexo-renderer-markdown-it，所以我卸载后者
npm install hexo-renderer-pandoc --save #安装新版本
```



#### 更新部署

可以看到对于复杂的公式也是支持的~

![image-20200813003737055](https://i.loli.net/2020/08/13/cXzadkQeCvAIwfB.png)



### 注意事项

如果你使用这款 Pandoc renderer，那么书写 Markdown 时候需要遵循 [Pandoc 对 Markdown 的规定](https://pandoc.org/MANUAL.html#pandocs-markdown)。

有一些比较明显的需要注意的事项：正常的文字后面如果跟的是[`list`](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#lists), [`table`](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables)或者[`quotation`](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#blockquotes)，文字后面需要空一行，如果不空行，这些环境将不能被 Pandoc renderer 正常渲染。



### 参考

> Hexo渲染Latex出现的问题：  https://zhuanlan.zhihu.com/p/35988761
>
> 