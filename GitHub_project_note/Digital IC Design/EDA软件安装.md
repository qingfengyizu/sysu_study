内容：EDA软件安装

时间：2020.7.20





## 1.安装红帽6.8

1 安装Redhat 6.8 版本系统，参考[安装教程](https://www.osyunwei.com/archives/7129.html)

2 安装 yum，参考了，[博客](https://blog.51cto.com/hyfcto/2125554)和[博客](https://zhuanlan.zhihu.com/p/29997446)

- 检查是否安装yum包

  ```
  rpm -qa |grep yum
  ```

- 删除自带的yum包

  ```
  rpm -qa|grep yum|xargs rpm -e --nodeps
  ```

- 下载yum包，Yum包下载地址http://mirrors.163.com/centos/，r**pm 版本会更新版本会更新**，进入 [http://mirrors.163.com/centos/6/os/x86_64/Packages/](http://mirrors.163.com/centos/6/os/x86_64/Packages/python-iniparse-0.3.1-2.1.el6.noarch.rpm)查找对应版本

  ```
  wget http://mirrors.163.com/centos/6/os/x86_64/Packages/python-iniparse-0.3.1-2.1.el6.noarch.rpm
  
  wget http://mirrors.163.com/centos/6/os/x86_64/Packages/yum-3.2.29-73.el6.centos.noarch.rpm
  
  wget http://mirrors.163.com/centos/6/os/x86_64/Packages/yum-metadata-parser-1.1.2-16.el6.x86_64.rpm
  
  wget http://mirrors.163.com/centos/6/os/x86_64/Packages/yum-plugin-fastestmirror-1.1.30-37.el6.noarch.rpm
  ```

- 解压yum包，**最后两个需要一起安装，有相互依赖关系**

  ```
  rpm -ivh python-iniparse-0.3.1-2.1.el6.noarch.rpm
  
  rpm -ivh yum-metadata-parser-1.1.2-16.el6.x86_64.rpm
  
  rpm -ivh yum-3.2.29-73.el6.centos.noarch.rpm yum-plugin-fastestmirror-1.1.30-37.el6.noarch.rpm
  ```

- 替换yum源

  ```
  cd /etc/yum.repos.d/
  
  wget http://mirrors.163.com/.help/CentOS6-Base-163.repo
  
  vi CentOS6-Base-163.repo
  
  mv rhel-source.repo rhel-source.repo.bak
  mv CentOS6-Base-163.repo rhel-source.repo
  ```

- 编辑修改rhel-source.repo

  ```
  ：%s/$releasever/6/g（在 vi 命令模式下执行上述命令）
  ```

- 清理并重建缓存

  ```
  yum clean all
  yum makecache
  ```

- 更新yum

  ```
  yum update
  ```

# 1 安装centos 6.8



## 2 gvim

```
# ------ summary ---------------------------------------------
vi is an powerful file editor for programming in Linux OS.
vim       : vi improved
gvim      : GUI of vi 
.vim      : highligh word file
.vimrc    : configuration file of VI

two mode  : editing(insert) and command mode

# ------ file operation -----------------------------------------

# ------ open a file -----------
vi file_name   : open a file for editing on a terminal
vim file_name  :
gvim file_name : gvim is a GUI of vi

# when open a file , vi is in insert mode by default

i     : go to insert mode 
esc    : go to command mode
:w     : write into the file (save)
:q     : quit vi 
:q!    : force to quit and abort the modification
:wq    : save and quit

# ----- move cursor ------------
->/<-     : left/right/up/down 
h|j|k|l   : h(left)| j(down) | k(up) l (right)
          : 3h | 4j | 5k | 6l
:w        : move forward one word eg. 3w
:b        : move backward one word eg. 4b
:$        : move to the end of a line
:^|0      : move to the beginning of a line
# ---------------------------

gg          : go to the first line
G           : go to the last line
nG          : go to n line eg. 1G
:number     :
:set nu     : set number line
:set nonu
CTRL + G    : display the current line and total numbers of lines
CTRL + U    : page up
CTRL + D    : page down

# ----- delete copy and paste -----

d=delete, y=copy, p=paste

dd       : delete a line eg. 5dd
dw       : delete a word eg. d3w
d0       : 
d$       : delete to end of line

yy       : copy a line et. 5yy
yw 
y0 
y$ 
Y        : copy
:5,10y    : copy 5 ~ 10 line
:,10y     : copy cursor ~ 10 line
:5,y      : copy 5 ~ cursor line


:p        : paste

:.        : rpeat last operation 

:x        : dlete a character eg. 3x

# ------ undo the editing  ----
:u|U      : undo | redo
:CTRL+R   : ned the modification

# ------ inset cursor -------------------
a|A       : ater the cursor | end of a line
o|O       : input one new line under th*e current line | up the current line
:help a   : 

# ------ search  ---------------------
:/pattern : go to the pattern
          : n|N
:?pattern :
SHIFT + * : match the word marked cursor
:nohl
:number_line : go to the number line

# ----- replace -----------------------
:r|R          : replace
:%s/x/y/g     : x change to y all of them
:s/x/y/g      : x change to y on the current line
:10,23s/x/y/g : 

# ----- special operation ------------------
:sp       : splite horizontally ; put some files into one terminal
CRTL + WW : change file in splite command
:ZZ|q     : quit a splite file

:set diff : compare two files
:vsp      : visual splite vertically
CRTL + WW : 

CRTL + V  : visual mode
          : d|D , y|Y, r|R
SHIFT + I : insert mode for editing
ESC       : Mactch visual mode
          
gf        : go into file
CRTL + O  : return the original file 

# ------ other command -------------------
J        : Merge the under line and the current line eg. 3JA
~        : change case-sensitive character 

```

