---
title: 2020-08-08-next缺少custom.styl的问题
date: 2020-08-08 18:56:09
tags: [故障排除,next]
top:
categories: next
description: next7.x 版本没有custom.styl文件
---

### 前言

在 next7.x 版本中没有custom.styl文件。如果我们想要在博客中添加自己的css样式，可以在此文件中添加，下面介绍一下



### 操作

step1 ：添加custom.styl文件

文件路径：`~\themes\next\source\css` ,添加`_custom`文件夹。然后在`_custom`中创建`custom.styl`文件。我们自己的样式就可以在此文件中添加



step2： 添加引用

在`~\themes\next\source\css`中的`main.styl`文件末尾加入引用即可

```
//My Layer
@import "_custom/custom.styl";
```



step3： 添加样式

用vscode打开`custom.styl`，博客背景以及前页的不透明度等等，就可以更换样式了。

对于网页的组件，F12打开调试界面，就可以知道每个组件的名称等信息，便于更改样式









