[Python - 100天从新手到大师](https://github.com/jackfrued/Python-100-Days) 

### 1、linux 基础命令

Linux系统的命令通常都是如下所示的格式：

```shell
命令名称 [命名参数] [命令对象]
```

#### 查看自己使用的Shell - **ps**。

​		Shell也被称为“壳”或“壳程序”，它是用户与操作系统内核交流的翻译官，简单的说就是人与计算机交互的界面和接口。目前很多Linux系统默认的Shell都是bash（<u>B</u>ourne <u>A</u>gain <u>SH</u>ell），因为它可以使用tab键进行命令和路径补全、可以保存历史命令、可以方便的配置环境变量以及执行批处理操作。

```shell
[root@izwz97tbgo9lkabnat2lo8z ~]# ps
  PID TTY          TIME CMD
 3531 pts/0    00:00:00 bash
 3553 pts/0    00:00:00 ps	
```



#### 查看命令的说明和位置 - **whatis** / **which** / **whereis**。

​		

```shell
[root ~]# whatis ps
ps (1)        - report a snapshot of the current processes.
[root ~]# whatis python
python (1)    - an interpreted, interactive, object-oriented programming language
[root ~]# whereis ps
ps: /usr/bin/ps /usr/share/man/man1/ps.1.gz
[root ~]# whereis python
python: /usr/bin/python /usr/bin/python2.7 /usr/lib/python2.7 /usr/lib64/python2.7 /etc/python /usr/include/python2.7 /usr/share/man/man1/python.1.gz
[root ~]# which ps
/usr/bin/ps
[root ~]# which python
/usr/bin/python
```



#### 查看帮助文档 - **man** / **info** / **help** / **apropos**。



#### 查看系统和主机名 - **uname** / **hostname**。

```shell
[root@izwz97tbgo9lkabnat2lo8z ~]# uname
Linux
[root@izwz97tbgo9lkabnat2lo8z ~]# hostname
izwz97tbgo9lkabnat2lo8z
[root@iZwz97tbgo9lkabnat2lo8Z ~]# cat /etc/centos-release
CentOS Linux release 7.6.1810 (Core)
```





### 2、基本操作

#### 文件和文件夹操作

​		创建/删除空目录 - **mkdir** / **rmdir**。

​		创建/删除文件 - **touch** / **rm**。

​		查看文件内容 - **cat** / **tac** / **head** / **tail** / **more** / **less** / **rev** / **od**。

​		拷贝/移动文件 - **cp** / **mv**。

​		文件重命名 - **rename**。

​		查找文件和查找内容 - **find** / **grep**。`grep`在搜索字符串时可以使用正则表达式，如果需要使用正则表达式可以用`grep -E`或者直接使用`egrep`。

​		创建链接和查看链接 - **ln** / **readlink**。

​		压缩/解压缩和归档/解归档 - **gzip** / **gunzip** / **xz**。

​		归档和解归档 - **tar**。

​		其他相关工具。 

   - **sort** - 对内容排序
   - **uniq** - 去掉相邻重复内容
   - **tr** - 替换指定内容为新内容
   - **cut** / **paste** - 剪切/黏贴内容
   - **split** - 拆分文件
   - **file** - 判断文件类型
   - **wc** - 统计文件行数、单词数、字节数
   - **iconv** - 编码转换

#### 管道和重定向

1. 管道的使用 - **\|**。

2. 输出重定向和错误重定向 - **\>** / **>>** / **2\>**。

3. 输入重定向 - **\<**。

4. 多重定向 - **tee**。
	下面的命令除了在终端显示命令`ls`的结果之外，还会追加输出到`ls.txt`文件中。
	
	```shell
	[root ~]# ls | tee -a ls.txt
	```
	
	

#### 别名

1. **alias**

   ```Shell
   [root ~]# alias ll='ls -l'
   [root ~]# alias frm='rm -rf'
   [root ~]# ll
   ...
   drwxr-xr-x  2 root       root   4096 Jun 20 12:52 abc
   ...
   [root ~]# frm abc
   ```

2. **unalias**

   ```Shell
   [root ~]# unalias frm
   [root ~]# frm sohu.html
   -bash: frm: command not found
   ```

#### 文本处理

1. 字符流编辑器 - **sed**。

   sed是操作、过滤和转换文本内容的工具。假设有一个名为fruit.txt的文件，内容如下所示。

   ```Shell
   [root ~]# cat -n fruit.txt 
        1  banana
        2  grape
        3  apple
        4  watermelon
        5  orange
   ```

   接下来，我们在第2行后面添加一个pitaya。

   ```Shell
   [root ~]# sed '2a pitaya' fruit.txt 
   banana
   grape
   pitaya
   apple
   watermelon
   orange
   ```

   > 注意：刚才的命令和之前我们讲过的很多命令一样并没有改变fruit.txt文件，而是将添加了新行的内容输出到终端中，如果想保存到fruit.txt中，可以使用输出重定向操作。

   在第2行前面插入一个waxberry。

   ```Shell
   [root ~]# sed '2i waxberry' fruit.txt
   banana
   waxberry
   grape
   apple
   watermelon
   orange
   ```

   删除第3行。

   ```Shell
   [root ~]# sed '3d' fruit.txt
   banana
   grape
   watermelon
   orange
   ```

   删除第2行到第4行。

   ```Shell
   [root ~]# sed '2,4d' fruit.txt
   banana
   orange
   ```

   将文本中的字符a替换为@。

   ```Shell
   [root ~]# sed 's#a#@#' fruit.txt 
   b@nana
   gr@pe
   @pple
   w@termelon
   or@nge
   ```

   将文本中的字符a替换为@，使用全局模式。

   ```Shell
   [root ~]# sed 's#a#@#g' fruit.txt 
   b@n@n@
   gr@pe
   @pple
   w@termelon
   or@nge
   ```

2. 模式匹配和处理语言 - **awk**。

   awk是一种编程语言，也是Linux系统中处理文本最为强大的工具，它的作者之一和现在的维护者就是之前提到过的Brian Kernighan（ken和dmr最亲密的伙伴）。通过该命令可以从文本中提取出指定的列、用正则表达式从文本中取出我们想要的内容、显示指定的行以及进行统计和运算，总之它非常强大。

   假设有一个名为fruit2.txt的文件，内容如下所示。

   ```Shell
   [root ~]# cat fruit2.txt 
   1       banana      120
   2       grape       500
   3       apple       1230
   4       watermelon  80
   5       orange      400
   ```

   显示文件的第3行。

   ```Shell
   [root ~]# awk 'NR==3' fruit2.txt 
   3       apple       1230
   ```

   显示文件的第2列。

   ```Shell
   [root ~]# awk '{print $2}' fruit2.txt 
   banana
   grape
   apple
   watermelon
   orange
   ```

   显示文件的最后一列。

   ```Shell
   [root ~]# awk '{print $NF}' fruit2.txt 
   120
   500
   1230
   80
   400
   ```

   输出末尾数字大于等于300的行。

   ```Shell
   [root ~]# awk '{if($3 >= 300) {print $0}}' fruit2.txt 
   2       grape       500
   3       apple       1230
   5       orange      400
   ```

   上面展示的只是awk命令的冰山一角，更多的内容留给读者自己在实践中去探索。

   
   
   
   
   
   
   

#### 用户管理

1. 创建和删除用户 - **useradd** / **userdel**。

   ```Shell
   [root home]# useradd hellokitty
   [root home]# userdel hellokitty
   ```

   - `-d` - 创建用户时为用户指定用户主目录
   - `-g` - 创建用户时指定用户所属的用户组

2. 创建和删除用户组 - **groupadd** / **groupdel**。

   > 说明：用户组主要是为了方便对一个组里面所有用户的管理。

3. 修改密码 - **passwd**。

   ```Shell
   [root ~]# passwd hellokitty
   New password: 
   Retype new password: 
   passwd: all authentication tokens updated successfully.
   ```

   > 说明：输入密码和确认密码没有回显且必须一气呵成的输入完成（不能使用退格键），密码和确认密码需要一致。如果使用`passwd`命令时没有指定命令作用的对象，则表示要修改当前用户的密码。如果想批量修改用户密码，可以使用`chpasswd`命令。

   - `-l` / `-u` - 锁定/解锁用户。
   - `-d` - 清除用户密码。
   - `-e` - 设置密码立即过期，用户登录时会强制要求修改密码。
   - `-i` - 设置密码过期多少天以后禁用该用户。

4. 切换用户 - **su**。

   ```Shell
   [root ~]# su hellokitty
   [hellokitty root]$
   ```

5. 以管理员身份执行命令 - **sudo**。

   ```Shell
   [hellokitty ~]$ ls /root
   ls: cannot open directory /root: Permission denied
   [hellokitty ~]$ sudo ls /root
   [sudo] password for hellokitty:
   ```

   > **说明**：如果希望用户能够以管理员身份执行命令，用户必须要出现在sudoers名单中，sudoers文件在 `/etc`目录下，如果希望直接编辑该文件也可以使用下面的命令。



#### 文件系统

##### 文件和路径

1. 命名规则：文件名的最大长度与文件系统类型有关，一般情况下，文件名不应该超过255个字符，虽然绝大多数的字符都可以用于文件名，但是最好使用英文大小写字母、数字、下划线、点这样的符号。文件名中虽然可以使用空格，但应该尽可能避免使用空格，否则在输入文件名时需要用将文件名放在双引号中或者通过`\`对空格进行转义。
2. 扩展名：在Linux系统下文件的扩展名是可选的，但是使用扩展名有助于对文件内容的理解。有些应用程序要通过扩展名来识别文件，但是更多的应用程序并不依赖文件的扩展名，就像`file`命令在识别文件时并不是依据扩展名来判定文件的类型。
3. 隐藏文件：以点开头的文件在Linux系统中是隐藏文件（不可见文件）。

##### 目录结构

1. /bin - 基本命令的二进制文件。
2. /boot - 引导加载程序的静态文件。
3. /dev - 设备文件。
4. **/etc** - 配置文件。
5. /home - 普通用户主目录的父目录。
6. /lib - 共享库文件。
7. /lib64 - 共享64位库文件。
8. /lost+found - 存放未链接文件。
9. /media - 自动识别设备的挂载目录。
10. /mnt - 临时挂载文件系统的挂载点。
11. /opt - 可选插件软件包安装位置。
12. /proc -  内核和进程信息。
13. **/root** - 超级管理员用户主目录。
14. /run - 存放系统运行时需要的东西。
15. /sbin - 超级用户的二进制文件。
16. /sys - 设备的伪文件系统。
17. /tmp - 临时文件夹。
18. **/usr** - 用户应用目录。
19. /var - 变量数据目录。
##### 访问权限

1. **chmod** - 改变文件模式比特。

##### 磁盘管理

1. 列出文件系统的磁盘使用状况 - **df**。

   ```Shell
   [root ~]# df -h
   Filesystem      Size  Used Avail Use% Mounted on
   /dev/vda1        40G  5.0G   33G  14% /
   devtmpfs        486M     0  486M   0% /dev
   tmpfs           497M     0  497M   0% /dev/shm
   tmpfs           497M  356K  496M   1% /run
   tmpfs           497M     0  497M   0% /sys/fs/cgroup
   tmpfs           100M     0  100M   0% /run/user/0
   ```

2. 磁盘分区表操作 - **fdisk**。

   ```Shell
   [root ~]# fdisk -l
   Disk /dev/vda: 42.9 GB, 42949672960 bytes, 83886080 sectors
   Units = sectors of 1 * 512 = 512 bytes
   Sector size (logical/physical): 512 bytes / 512 bytes
   I/O size (minimum/optimal): 512 bytes / 512 bytes
   Disk label type: dos
   Disk identifier: 0x000a42f4
      Device Boot      Start         End      Blocks   Id  System
   /dev/vda1   *        2048    83884031    41940992   83  Linux
   Disk /dev/vdb: 21.5 GB, 21474836480 bytes, 41943040 sectors
   Units = sectors of 1 * 512 = 512 bytes
   Sector size (logical/physical): 512 bytes / 512 bytes
   I/O size (minimum/optimal): 512 bytes / 512 bytes
   ```

3. 磁盘分区工具 - **parted**。

4. 格式化文件系统 - **mkfs**。

   ```Shell
   [root ~]# mkfs -t ext4 -v /dev/sdb
   ```

   - `-t` - 指定文件系统的类型。
   - `-c` - 创建文件系统时检查磁盘损坏情况。
   - `-v` - 显示详细信息。

5. 文件系统检查 - **fsck**。

6. 转换或拷贝文件 - **dd**。

7. 挂载/卸载 - **mount** / **umount**。

8. 创建/激活/关闭交换分区 - **mkswap** / **swapon** / **swapoff**。

> 说明：执行上面这些命令会带有一定的风险，如果不清楚这些命令的用法，最好不用随意使用，在使用的过程中，最好对照参考资料进行操作，并在操作前确认是否要这么做。

#### 编辑器 - vim

1. 启动vim。可以通过`vi`或`vim`命令来启动vim，启动时可以指定文件名来打开一个文件，如果没有指定文件名，也可以在保存的时候指定文件名。

   ```Shell
   [root ~]# vim guess.py
   ```

2. 命令模式、编辑模式和末行模式：启动vim进入的是命令模式（也称为Normal模式），在命令模式下输入英文字母`i`会进入编辑模式（Insert模式），屏幕下方出现`-- INSERT --`提示；在编辑模式下按下`Esc`会回到命令模式，此时如果输入英文`:`会进入末行模式，在末行模式下输入`q!`可以在不保存当前工作的情况下强行退出vim；在命令模式下输入`v`会进入可视模式（Visual模式），可以用光标选择一个区域再完成对应的操作。

3. 保存和退出vim：在命令模式下输入`:` 进入末行模式，输入`wq`可以实现保存退出；如果想放弃编辑的内容输入`q!`强行退出，这一点刚才已经提到过了；在命令模式下也可以直接输入`ZZ`实现保存退出。如果只想保存文件不退出，那么可以在末行模式下输入`w`；可以在`w`后面输入空格再指定要保存的文件名。

4. 光标操作。

   - 在命令模式下可以通过`h`、`j`、`k`、`l`来控制光标向左、下、上、右的方向移动，可以在字母前输入数字来表示移动的距离，例如：`10h`表示向左移动10个字符。
   - 在命令模式下可以通过`Ctrl+y`和`Ctrl+e`来实现向上、向下滚动一行文本的操作，可以通过`Ctrl+f`和`Ctrl+b`来实现向前和向后翻页的操作。
   - 在命令模式下可以通过输入英文字母`G`将光标移到文件的末尾，可以通过`gg`将光标移到文件的开始，也可以通过在`G`前输入数字来将光标移动到指定的行。

5. 文本操作。

   - 删除：在命令模式下可以用`dd`来删除整行；可以在`dd`前加数字来指定删除的行数；可以用`d$`来实现删除从光标处删到行尾的操作，也可以通过`d0`来实现从光标处删到行首的操作；如果想删除一个单词，可以使用`dw`；如果要删除全文，可以在输入`:%d`（其中`:`用来从命令模式进入末行模式）。
   - 复制和粘贴：在命令模式下可以用`yy`来复制整行；可以在`yy`前加数字来指定复制的行数；可以通过`p`将复制的内容粘贴到光标所在的地方。
   - 撤销和恢复：在命令模式下输入`u`可以撤销之前的操作；通过`Ctrl+r`可以恢复被撤销的操作。
   - 对内容进行排序：在命令模式下输入`%!sort`。

6. 查找和替换。

   - 查找操作需要输入`/`进入末行模式并提供正则表达式来匹配与之对应的内容，例如：`/doc.*\.`，输入`n`来向前搜索，也可以输入`N`来向后搜索。
   - 替换操作需要输入`:`进入末行模式并指定搜索的范围、正则表达式以及替换后的内容和匹配选项，例如：`:1,$s/doc.*/hello/gice`，其中：
     - `g` - global：全局匹配。
     - `i` - ignore case：忽略大小写匹配。
     - `c` - confirm：替换时需要确认。
     - `e` - error：忽略错误。

7. 参数设定：在输入`:`进入末行模式后可以对vim进行设定。

   - 设置Tab键的空格数：`set ts=4`

   - 设置显示/不显示行号：`set nu` / `set nonu`

   - 设置启用/关闭高亮语法：`syntax on` / `syntax off`

   - 设置显示标尺（光标所在的行和列）： `set ruler`

   - 设置启用/关闭搜索结果高亮：`set hls` / `set nohls`

     > 说明：如果希望上面的这些设定在每次启动vim时都能自动生效，需要将这些设定写到用户主目录下的.vimrc文件中。

#### 软件安装和配置


1. **yum** - Yellowdog Updater Modified。
   - `yum search`：搜索软件包，例如`yum search nginx`。
   - `yum list installed`：列出已经安装的软件包，例如`yum list installed | grep zlib`。
   - `yum install`：安装软件包，例如`yum install nginx`。
   - `yum remove`：删除软件包，例如`yum remove nginx`。
   - `yum update`：更新软件包，例如`yum update`可以更新所有软件包，而`yum update tar`只会更新tar。
   - `yum check-update`：检查有哪些可以更新的软件包。
   - `yum info`：显示软件包的相关信息，例如`yum info nginx`。
2. **rpm** - Redhat Package Manager。
   - 安装软件包：`rpm -ivh <packagename>.rpm`。
   - 移除软件包：`rpm -e <packagename>`。
   - 查询软件包：`rpm -qa`，例如可以用`rpm -qa | grep mysql`来检查是否安装了MySQL相关的软件包。

### 网络访问和管理

1. 安全远程连接 - **ssh**。

   ```Shell
   [root ~]$ ssh root@120.77.222.217
   The authenticity of host '120.77.222.217 (120.77.222.217)' can't be established.
   ECDSA key fingerprint is SHA256:BhUhykv+FvnIL03I9cLRpWpaCxI91m9n7zBWrcXRa8w.
   ECDSA key fingerprint is MD5:cc:85:e9:f0:d7:07:1a:26:41:92:77:6b:7f:a0:92:65.
   Are you sure you want to continue connecting (yes/no)? yes
   Warning: Permanently added '120.77.222.217' (ECDSA) to the list of known hosts.
   root@120.77.222.217's password: 
   ```

2. 通过网络获取资源 - **wget**。

   - -b 后台下载模式
   - -O 下载到指定的目录
   - -r 递归下载

3. 发送和接收邮件 - **mail**。

4. 网络配置工具（旧） - **ifconfig**。

   ```Shell
   [root ~]# ifconfig eth0
   eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
           inet 172.18.61.250  netmask 255.255.240.0  broadcast 172.18.63.255
           ether 00:16:3e:02:b6:46  txqueuelen 1000  (Ethernet)
           RX packets 1067841  bytes 1296732947 (1.2 GiB)
           RX errors 0  dropped 0  overruns 0  frame 0
           TX packets 409912  bytes 43569163 (41.5 MiB)
           TX errors 0  dropped 0 overruns 0  carrier 0  collisions 
   ```

5. 网络配置工具（新） - **ip**。

   ```Shell
   [root ~]# ip address
   1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN qlen 1
       link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
       inet 127.0.0.1/8 scope host lo
          valid_lft forever preferred_lft forever
   2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP qlen 1000
       link/ether 00:16:3e:02:b6:46 brd ff:ff:ff:ff:ff:ff
       inet 172.18.61.250/20 brd 172.18.63.255 scope global eth0
          valid_lft forever preferred_lft forever
   ```

6. 网络可达性检查 - **ping**。

   ```Shell
   [root ~]# ping www.baidu.com -c 3
   PING www.a.shifen.com (220.181.111.188) 56(84) bytes of data.
   64 bytes from 220.181.111.188 (220.181.111.188): icmp_seq=1 ttl=51 time=36.3 ms
   64 bytes from 220.181.111.188 (220.181.111.188): icmp_seq=2 ttl=51 time=36.4 ms
   64 bytes from 220.181.111.188 (220.181.111.188): icmp_seq=3 ttl=51 time=36.4 ms
   --- www.a.shifen.com ping statistics ---
   3 packets transmitted, 3 received, 0% packet loss, time 2002ms
   rtt min/avg/max/mdev = 36.392/36.406/36.427/0.156 ms
   ```

7. 显示或管理路由表 - **route**。

8. 查看网络服务和端口 - **netstat** / **ss**。

   ```Shell
   [root ~]# netstat -nap | grep nginx
   ```

9. 网络监听抓包 - **tcpdump**。

10. 安全文件拷贝 - **scp**。

  ```Shell
  [root ~]# scp root@1.2.3.4:/root/guido.jpg hellokitty@4.3.2.1:/home/hellokitty/pic.jpg
  ```

11. 文件同步工具 - **rsync**。

    > 说明：使用`rsync`可以实现文件的自动同步，这个对于文件服务器来说相当重要。关于这个命令的用法，我们在后面讲项目部署的时候为大家详细说明。

12. 安全文件传输 - **sftp**。

    ```Shell
    [root ~]# sftp root@1.2.3.4
    root@1.2.3.4's password:
    Connected to 1.2.3.4.
    sftp>
    ```

    - `help`：显示帮助信息。

    - `ls`/`lls`：显示远端/本地目录列表。

    - `cd`/`lcd`：切换远端/本地路径。

    - `mkdir`/`lmkdir`：创建远端/本地目录。

    - `pwd`/`lpwd`：显示远端/本地当前工作目录。

    - `get`：下载文件。

    - `put`：上传文件。

    - `rm`：删除远端文件。

    - `bye`/`exit`/`quit`：退出sftp。

### 

### 进程管理

1. 查看进程 - **ps**。

   ```Shell
   [root ~]# ps -ef
   UID        PID  PPID  C STIME TTY          TIME CMD
   root         1     0  0 Jun23 ?        00:00:05 /usr/lib/systemd/systemd --switched-root --system --deserialize 21
   root         2     0  0 Jun23 ?        00:00:00 [kthreadd]
   ...
   [root ~]# ps -ef | grep mysqld
   root      4943  4581  0 22:45 pts/0    00:00:00 grep --color=auto mysqld
   mysql    25257     1  0 Jun25 ?        00:00:39 /usr/sbin/mysqld --daemonize --pid-file=/var/run/mysqld/mysqld.pid
   ```

2. 显示进程状态树 - **pstree**。

   ```Shell
   [root ~]# pstree
   systemd─┬─AliYunDun───18*[{AliYunDun}]
           ├─AliYunDunUpdate───3*[{AliYunDunUpdate}]
           ├─2*[agetty]
           ├─aliyun-service───2*[{aliyun-service}]
           ├─atd
           ├─auditd───{auditd}
           ├─dbus-daemon
           ├─dhclient
           ├─irqbalance
           ├─lvmetad
           ├─mysqld───28*[{mysqld}]
           ├─nginx───2*[nginx]
           ├─ntpd
           ├─polkitd───6*[{polkitd}]
           ├─rsyslogd───2*[{rsyslogd}]
           ├─sshd───sshd───bash───pstree
           ├─systemd-journal
           ├─systemd-logind
           ├─systemd-udevd
           └─tuned───4*[{tuned}]
   ```

3. 查找与指定条件匹配的进程 - **pgrep**。

   ```Shell
   [root ~]$ pgrep mysqld
   3584
   ```

4. 通过进程号终止进程 - **kill**。

   ```Shell
   [root ~]$ kill -l
    1) SIGHUP       2) SIGINT       3) SIGQUIT      4) SIGILL       5) SIGTRAP
    6) SIGABRT      7) SIGBUS       8) SIGFPE       9) SIGKILL     10) SIGUSR1
   11) SIGSEGV     12) SIGUSR2     13) SIGPIPE     14) SIGALRM     15) SIGTERM
   16) SIGSTKFLT   17) SIGCHLD     18) SIGCONT     19) SIGSTOP     20) SIGTSTP
   21) SIGTTIN     22) SIGTTOU     23) SIGURG      24) SIGXCPU     25) SIGXFSZ
   26) SIGVTALRM   27) SIGPROF     28) SIGWINCH    29) SIGIO       30) SIGPWR
   31) SIGSYS      34) SIGRTMIN    35) SIGRTMIN+1  36) SIGRTMIN+2  37) SIGRTMIN+3
   38) SIGRTMIN+4  39) SIGRTMIN+5  40) SIGRTMIN+6  41) SIGRTMIN+7  42) SIGRTMIN+8
   43) SIGRTMIN+9  44) SIGRTMIN+10 45) SIGRTMIN+11 46) SIGRTMIN+12 47) SIGRTMIN+13
   48) SIGRTMIN+14 49) SIGRTMIN+15 50) SIGRTMAX-14 51) SIGRTMAX-13 52) SIGRTMAX-12
   53) SIGRTMAX-11 54) SIGRTMAX-10 55) SIGRTMAX-9  56) SIGRTMAX-8  57) SIGRTMAX-7
   58) SIGRTMAX-6  59) SIGRTMAX-5  60) SIGRTMAX-4  61) SIGRTMAX-3  62) SIGRTMAX-2
   63) SIGRTMAX-1  64) SIGRTMAX
   [root ~]# kill 1234
   [root ~]# kill -9 1234
   ```

   例子：用一条命令强制终止正在运行的Redis进程。

    ```Shell
   ps -ef | grep redis | grep -v grep | awk '{print $2}' | xargs kill
    ```

5. 通过进程名终止进程 - **killall** / **pkill**。

   结束名为mysqld的进程。

   ```Shell
   [root ~]# pkill mysqld
   ```

   结束hellokitty用户的所有进程。

   ```Shell
   [root ~]# pkill -u hellokitty
   ```

   > 说明：这样的操作会让hellokitty用户和服务器断开连接。

6. 将进程置于后台运行。

   - `Ctrl+Z` - 快捷键，用于停止进程并置于后台。
   - `&` - 将进程置于后台运行。

   ```Shell
   [root ~]# mongod &
   [root ~]# redis-server
   ...
   ^Z
   [4]+  Stopped                 redis-server
   ```

7. 查询后台进程 - **jobs**。

   ```Shell
   [root ~]# jobs
   [2]   Running                 mongod &
   [3]-  Stopped                 cat
   [4]+  Stopped                 redis-server
   ```

8. 让进程在后台继续运行 - **bg**。

   ```Shell
   [root ~]# bg %4
   [4]+ redis-server &
   [root ~]# jobs
   [2]   Running                 mongod &
   [3]+  Stopped                 cat
   [4]-  Running                 redis-server &
   ```

9. 将后台进程置于前台 - **fg**。

   ```Shell
   [root ~]# fg %4
   redis-server
   ```

   > 说明：置于前台的进程可以使用`Ctrl+C`来终止它。

10. 调整程序/进程运行时优先级 - **nice** / **renice**。

11. 用户登出后进程继续工作 - **nohup**。

     ```Shell
     [root ~]# nohup ping www.baidu.com > result.txt &
     ```

12. 跟踪进程系统调用情况 - **strace**。

     ```Shell
     [root ~]# pgrep mysqld
     8803
     [root ~]# strace -c -p 8803
     strace: Process 8803 attached
     ^Cstrace: Process 8803 detached
     % time     seconds  usecs/call     calls    errors syscall
     ------ ----------- ----------- --------- --------- ----------------
      99.18    0.005719        5719         1           restart_syscall
       0.49    0.000028          28         1           mprotect
       0.24    0.000014          14         1           clone
       0.05    0.000003           3         1           mmap
       0.03    0.000002           2         1           accept
     ------ ----------- ----------- --------- --------- ----------------
     100.00    0.005766                     5           total
     ```

     > 说明：这个命令的用法和参数都比较复杂，建议大家在真正用到这个命令的时候再根据实际需要进行了解。

13. 查看当前运行级别 - **runlevel**。

     ```Shell
     [root ~]# runlevel
     N 3
     ```

14. 实时监控进程占用资源状况 - **top**。

     ```Shell
     [root ~]# top
     top - 23:04:23 up 3 days, 14:10,  1 user,  load average: 0.00, 0.01, 0.05
     Tasks:  65 total,   1 running,  64 sleeping,   0 stopped,   0 zombie
     %Cpu(s):  0.3 us,  0.3 sy,  0.0 ni, 99.3 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
     KiB Mem :  1016168 total,   191060 free,   324700 used,   500408 buff/cache
     KiB Swap:        0 total,        0 free,        0 used.   530944 avail Mem
     ...
     ```

     - `-c` - 显示进程的整个路径。
     - `-d` - 指定两次刷屏之间的间隔时间（秒为单位）。
     - `-i` - 不显示闲置进程或僵尸进程。
     - `-p` - 显示指定进程的信息。

