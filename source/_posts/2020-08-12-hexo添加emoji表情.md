---
title: 2020-08-12-hexo添加emoji表情
mathjax: true
date: 2020-08-12 17:16:54
tags: [hexo,故障排除]
top: 
categories: hexo
description: 解决hexo无法渲染emoji表情的问题
---

### 前言

markdown支持在文本中使用emoji，在Typora中可以很方便地使用表情。例如输入 `:star:`  ,可以显示出:star:表情，即表情的`aliases` 编码格式。但是在部署到网站的时候，:star:却渲染不出来，我寻找了很久的解决方案，终于解决 :laughing:



### 更换hexo渲染器 

我的hexo版本是version 4.2.1, 可以在在根目录下 packge.json 文件里面看到使用hexo初始化的结果。

将markdown 变成html的转换器叫做`markdown渲染器`.在Hexo中默认的markdown渲染器 使用的是[hexo-renderer-marked](https://github.com/hexojs/hexo-renderer-marked),是Hexo版本，这个渲染器不支持插件扩展。另外一个 markdown 渲染器 [hexo-renderer-markdown-it](https://github.com/celsomiranda/hexo-renderer-markdown-it)，这个支持插件配置，可以使用 [markwon-it-emoji](https://github.com/markdown-it/markdown-it-emoji)插件来支持emoji。

解决方案：将原来的 `marked` 渲染器换成 `markdown-it`渲染器。



#### 安装新的渲染器

首先进入博客目录,卸载hexo默认的`marked`渲染器，安装`markdown-it`渲染器，运行的命令如：

```shell
cd Documents/blog
npm un hexo-renderer-marked --save  #卸载旧的渲染器
npm i hexo-renderer-markdown-it --save #暗转新的渲染器
```

之后安装`markdown-it-emoji`插件：

```shell
npm install markdown-it-emoji --save  
```



#### 编辑站点配置文件

这里的站点配置文件是指位于博客根目录下的 `_config.yml`，编辑它，然后在末尾添加如下内容：

```yml
# Markdown-it config
## Docs: https://github.com/celsomiranda/hexo-renderer-markdown-it/wiki
markdown:
  render:
    html: true
    xhtmlOut: false
    breaks: true
    linkify: true
    typographer: true
    quotes: '“”‘’'
  plugins:
    - markdown-it-abbr
    - markdown-it-footnote
    - markdown-it-ins
    - markdown-it-sub
    - markdown-it-sup
    - markdown-it-emoji  # add emoji
  anchors:
    level: 2
    collisionSuffix: 'v'
    permalink: true
    permalinkClass: header-anchor
    permalinkSymbol: ¶
```

上面的是`hexo-renderer-markdown-it`的所有选项的配置，详细的每一项配置说明，需要到[Advanced Configuration](https://github.com/celsomiranda/hexo-renderer-markdown-it/wiki/Advanced-Configuration)中查看。

这个时候就可以用表情的`aliases` 编码格式啦

如果觉得表情渲染的不好看，那么可以安装[twemoji](https://github.com/twitter/twemoji)，对表情进行优化。但是我对现在的渲染感到满意，就没有继续安装。



#### 查找emoji

表情的`aliases` 编码可以参考[emoji-cheat-sheet](https://www.webfx.com/tools/emoji-cheat-sheet/)，表情很全，可以找到每个表情的表示，运用到自己的文章里  



### Unicode编码方案

如果不更换hexo渲染器，那么可以使用表情的`Unicode`表达方式。不过不推荐此方式，感觉过于麻烦

语法：  `&#xCODE ;`

其中`CODE`是每个表情的编码方式，可以通过 [Emoji Unicode Tables](https://link.zhihu.com/?target=https%3A//apps.timwhitlock.info/emoji/tables/unicode%23block-4-enclosed-characters)查询得到

**例子：** 查到了 表情对应的 **Unicode** 编码为 `U+1F34E`，则与此表情对应的 `CODE` 为 `1F34E` (舍弃前面的 **U+**)。输入markdown文档内即可





### 后续

因为要读论文，然而在论文中会出现很多的数学公式，这时候需要运用Latex，原始的hexo渲染器[hexo-renderer-marked](https://github.com/hexojs/hexo-renderer-marked)对渲染不了公式，在为了能够添加emoji而更换的新渲染器 [hexo-renderer-markdown-it](https://github.com/celsomiranda/hexo-renderer-markdown-it)还是无法渲染Latex公式:neutral_face:

于是我又准备更换hexo渲染器，来让新的渲染器支持数学公式，于是我就更换了 [hexo-renderer-pandoc](https://github.com/wzpan/hexo-renderer-pandoc)，支持Mathjax语法，十分靠谱，然而问题来了，那就是不支持emoji了 :neutral_face:

在我准备在两者中舍弃一个，或者用emoji的Unicode编码来代替渲染器的时候，我发现了一个插件，就尝试在现有的渲染器基础上添加了一个hexo插件 [hexo-filter-github-emojis](https://github.com/crimx/hexo-filter-github-emojis) ，发现此插件可以有效支持emoji表情，于是两全其美啦:laughing:

------

下面是插件的使用说明

#### 安装插件

使用以下命令安装 [hexo-filter-github-emojis](https://github.com/crimx/hexo-filter-github-emojis) 插件：

```shell
 npm install hexo-filter-github-emojis --save
```

#### 启用插件

向站点配置文件 `hexo_root\_config.yml` 中添加如下设置：

```yml
githubEmojis:
  enable: true
  className: github-emoji
  unicode: true
  styles:
    display: inline
    vertical-align: middle # Freemind适用
  localEmojis:
```



具体的每个配置项含义参见 [说明文档](https://github.com/crimx/hexo-filter-github-emojis)。

#### 使用方法

和上述使用方法一样，很方便！ :rainbow:




### 参考

> hexo中添加表情： https://www.cnblogs.com/fsong/p/5929773.html
>
> hexo 使用emoji： https://spacefan.github.io/2018/06/30/hexo-emoji/