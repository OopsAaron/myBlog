---
title: 2020-09-28-windows-terminal探索
mathjax: true
date: 2020-09-28 10:38:58
tags:
top:
categories:
description: 才考完优化方法，折腾一下微软神器windows terminal
---

### 前言

优化方法终于考完了，最近几天都是在复习。可以放松两天，折腾一下



### windows terminal



#### 安装Windows Terminal



在Windows上，可以安装`Windows Terminal`。有点类似于MacOS上的`iTerm`，可以说是Windows下最舒适的终端。

在`windows store`中安装即可，比较方便，如下图所示

![image-20200930205359665](https://i.loli.net/2020/09/30/KLziDFhWNBg9qYR.png)





### 安装Ubuntu子系统

此时，我们仅仅安装了一个命令行终端而已，还是需要在`Windows`上安装`Ubuntu`。

只需要两步

1.在系统中开启子系统功能

2.在`windows store`安装linux版本即可。我安装的是`ubuntu 18.04 LTS`版本的，和实验室服务器一个版本号，利于操作。



> 关于LTS
>
> 1.LTS= 长期支持版本，你会在较长的时间内获得安全、维护和(有时有)功能的更新。
>
> 2.LTS 版本被认为是最稳定的版本，它经历了广泛的测试，并且大多包含了多年积累的改进。
>
> 3.对于ubuntu，没两年发布一个LST版本，比如ubuntu 16.04  ubuntu 18.04等等，代表的是发布的年份。
>
> 4.最新的 LTS 版本是 Ubuntu 20.04 ，它将被支持到 2025 年 4 月，支持5年的软件更新和修补。换句话说，Ubuntu 20.04 在那之前都会收到软件更新。非 LTS 版本只支持九个月。



如下图，在控制面板，找到程序选项，点击  “启用或关闭Windows功能”。

![image-20200928105917992](https://i.loli.net/2020/09/28/KQ3LocsNrBxiRf5.png)

从弹出的对话框里，划到最下边，然后给“适用于Linux的Windows子系统“，打勾，完事！

![image-20200928105907117](https://i.loli.net/2020/09/28/HPVhtscoGj8iCRA.png)



在windows中访问ubuntu系统

可以认为在windows 文件资源管理器中开辟一个空间用来储存ubuntu系统，但是如何找到位置呢？

执行如下命令：

```bash
cd /home  
explorer.exe .  #用文件资源管理器来打开当前home目录所在位置
```



可以看到是在`网络`一栏中， 可以看到ubuntu的文件目录。但是返回到`网络`根目录，却显示是无文件夹。不知道为什么。 

为了操作方便，我把这个长长的目录，右键映射到了Z盘上。如图，下次在访问Linux的时候，直接访问Z盘就可以了。

![image-20200928111036904](https://i.loli.net/2020/09/28/XqMOBfSKRkwHC93.png)



这时，就可以看到在我的电脑里就有了Z盘

![image-20200928111104530](https://i.loli.net/2020/09/28/V9MFRujrqgl46me.png)

>  **映射网络驱动器**目的就是为了让远程网络中的资源和本地共享，在本地可以对远程网络中的资源进行访问，并且可以创建文件。





### 工作区快捷键



| Win 快捷键               | 作用                                          | 备注                 |
| :----------------------- | :-------------------------------------------- | :------------------- |
| **Ctrl + Shift + P**，F1 | 显示命令面板                                  |                      |
| **Ctrl + B**             | 显示/隐藏侧边栏                               | 很实用               |
| `Ctrl + \`               | **创建多个编辑器**                            | 【重要】抄代码利器   |
| **Ctrl + 1、2**          | 聚焦到第 1、第 2 个编辑器                     | 同上重要             |
| **ctrl +/-**             | 将工作区放大/缩小（包括代码字体、左侧导航栏） | 在投影仪场景经常用到 |
| Ctrl + J                 | 显示/隐藏控制台                               |                      |
| **Ctrl + Shift + N**     | 重新开一个软件的窗口                          | 很常用               |
| Ctrl + Shift + W         | 关闭软件的当前窗口                            |                      |
| Ctrl + N                 | 新建文件                                      |                      |
| Ctrl + W                 | 关闭当前文件                                  |                      |