### linux 环境开发

#### 远程终端任务保持运行 tmux
一般都是在自己的工作电脑上安装远程连接工具（如 iTerm、putty、XShell、SecureCRT 等），远程登录到公司服务器上，进行具体的操作，而其中一些操作的耗时会很长。在这期间，一旦我们的远程连接工具所在的工作电脑出现断网或断电的情况，那么很多耗时较长的操作就会因此中断，这是所有运维、开发同学都很头疼的一个问题。

采用tmx软件来解决这个问题。

##### 安装tmux

``` linuxscript
apt-get install tmux
yum install tmux
```
##### 启动tmux

``` javascript
tmux new -s roclinux
```
-s是 session 的缩写，顾名思义，我们启动了一个全新的 tmux 会话（tmux session），并且把这个会话起名叫作 roclinux。这时，映入大家眼帘的就是 tmux 环境了。

##### 再创建一个新的窗口

 - 第一步：按 Ctrl+B 组合键，然后松开。
 - 第二步：再单独按一下 c 键

##### 在窗口间切换

 - 第一步：按 Ctrl-B 组合键，然后松开。
 - 第二步：按数字 0 键。

##### 回复端口

``` javascript
tmux ls
tmux a -t roclinux
```

#### 彻底关闭tmux窗口

``` javascript
tmux kill-window -t 16
```

参考：[tmux命令_Linux tmux命令：一个窗口操作多个会话](http://c.biancheng.net/linux/tmux.html)

