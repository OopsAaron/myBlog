---
title: 2020-07-27-linux的session解析
date: 2020-07-27 20:43:11
tags: linux
categories: linux
description: 关于ssh连接终端和session
---

### 前言

我们使用ssh连接服务器时，ssh的窗口突然断开了连接，那么在服务器上跑的程序就也跟着断掉了，之前所有跑的数据也将丢失，这样将会浪费我们大量的时间。



### 为什么ssh一旦断开我们的进程也将会被杀掉？

元凶：SIGHUP 信号

让我们来看看为什么关掉窗口/断开连接会使得正在运行的程序死掉。

在Linux/Unix中，有这样几个概念：

进程组（process group）：一个或多个进程的集合，每一个进程组有唯一一个进程组ID，即进程组长进程的ID。

会话期（session）：一个或多个进程组的集合，有唯一一个会话期首进程（session leader）。会话期ID为首进程的ID。

会话期可以有一个单独的控制终端（controlling terminal）。与控制终端连接的会话期首进程叫做控制进程（controlling process）。当前与终端交互的进程称为前台进程组。其余进程组称为后台进程组。

根据POSIX.1定义：

挂断信号（SIGHUP）默认的动作是终止程序。

当终端接口检测到网络连接断开，将挂断信号发送给控制进程（会话期首进程）。

如果会话期首进程终止，则该信号发送到该会话期前台进程组。

一个进程退出导致一个孤儿进程组中产生时，如果任意一个孤儿进程组进程处于STOP状态，发送SIGHUP和SIGCONT信号到该进程组中所有进程。

因此当网络断开或终端窗口关闭后，控制进程收到SIGHUP信号退出，会导致该会话期内其他进程退出。

**这里我认为我们的进程被杀掉也就是因为ssh与服务器之间的通信断掉了，这个通信断掉之后linux程序就默认将该连接下的所有进程都杀掉**



### session 是什么？

我们常见的 Linux session 一般是指 shell session。Shell session 是终端中当前的状态，在终端中只能有一个 session。`当我们打开一个新的终端时，总会创建一个新的 shell session。`

就进程间的关系来说，session 由一个或多个进程组组成。一般情况下，来自单个登录的所有进程都属于同一个 session。我们可以通过下图来理解进程、进程组和 session 之间的关系：

![img](https://img2018.cnblogs.com/blog/952033/202001/952033-20200103182042686-2100862807.png)

`会话是由会话中的第一个进程创建的，一般情况下是打开终端时创建的 shell 进程。`该进程也叫 session 的领头进程。Session 中领头进程的 PID 也就是 session 的 SID。我们可以通过下面的命令查看 SID：

```python
$ ps -o pid,ppid,pgid,sid,tty,comm
```

![img](https://img2018.cnblogs.com/blog/952033/202001/952033-20200103182117962-99639442.png)

Session 中的每个进程组被称为一个 job，有一个 job 会成为 session 的前台 job(foreground)，其它的 job 则是后台 job(background)。每个 session 连接一个控制终端(control terminal)，控制终端中的输入被发送给前台 job，从前台 job 产生的输出也被发送到控制终端上。同时由控制终端产生的信号，比如 ctrl + z 等都会传递给前台 job。

一般情况下 session 和终端是一对一的关系，当我们打开多个终端窗口时，实际上就创建了多个 session。

`Session 的意义在于多个工作(job)在一个终端中运行，其中的一个为前台 job，它直接接收该终端的输入并把结果输出到该终端。其它的 job 则在后台运行。`

### session 的诞生与消亡

通常，新的 session 由系统登录程序创建，session 中的领头进程是运行用户登录 shell 的进程。`新创建的每个进程都会属于一个进程组，当创建一个进程时，它和父进程在同一个进程组、session 中。`

将进程放入不同 session 的惟一方法是使用 setsid 函数使其成为新 session 的领头进程。这还会将 session 领头进程放入一个新的进程组中。

`当 session 中的所有进程都结束时 session 也就消亡了`。如下两种：

1.实际使用中比如网络断开了，session 肯定是要消亡的。

2.正常的消亡，比如让 session 的领头进程退出。

一般情况下 session 的领头进程是 shell 进程，如果它处于前台，我们可以使用 `exit 命令或者是 ctrl + d` 让它退出。或者我们可以直接通过 kill 命令杀死 session 的领头进程。这里面的原理是：当系统检测到挂断(hangup)条件时，内核中的驱动会将 SIGHUP 信号发送到整个 session。通常情况下，这会杀死 session 中的所有进程。

session 与终端的关系
如果 session 关联的是伪终端，这个伪终端本身就是随着 session 的建立而创建的，session 结束，那么这个伪终端也会被销毁。
如果 session 关联的是 tty1-6，tty 则不会被销毁。因为该终端设备是在系统初始化的时候创建的，并不是依赖该会话建立的，所以当 session 退出，tty 仍然存在。只是 init 系统在 session 结束后，会重启 getty 来监听这个 tty。

### nohup

`如果我们在 session 中执行了 nohup 等类似的命令，当 session 消亡时，相关的进程并不会随着 session 结束，原因是这些进程不再受 SIGHUP 信号的影响。`比如我们执行下面的命令：

```python
$ nohup sleep 1000 >/dev/null 2>&1 & 
```

![img](https://img2018.cnblogs.com/blog/952033/202001/952033-20200103182343352-1366915632.png)

此时 sleep 进程的 sid 和其它进程是相同的，还可以通过 pstree 命令看到进程间的父子关系：

![img](https://img2018.cnblogs.com/blog/952033/202001/952033-20200103182417115-817556079.png)

`如果我们退出当前 session 的领头进程(bash)，sleep 进程并不会退出，这样我们就可以放心的等待该进程运行结果了。`
nohup 并不改变进程的 sid，同时也说明在这种情况中，虽然 session 的领头进程退出了，但是 session 依然没有被销毁(至少 sid 还在被引用)。重新建立连接，通过下面的命令查看 sleep 进程的信息，发现进程的 sid 依然是 7837：

![img](https://img2018.cnblogs.com/blog/952033/202001/952033-20200103182448160-376880623.png)

但是`此时的 sleep 已经被系统的 1 号进程 systemd 收养了`：

![img](https://img2018.cnblogs.com/blog/952033/202001/952033-20200103182521953-1574746082.png)



### 参考

> https://www.cnblogs.com/sparkdev/p/12146305.html