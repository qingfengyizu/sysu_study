# 1. 基本16.04桌面系统安装
参考 https://blog.csdn.net/colin_lisicong/article/details/70193539

添加新用户

    useradd -r -m -s /bin/bash spark 
    passwd spark 



## 1.1解决tab无法补全
去除与tab相关的所有快捷键

    xfwm4-settings

利用vi编辑器打开/etc/bash.bashrc文件（需要root权限）


`sudo vi /etc/bash.bashrc ` 


找到文件中的下列代码


    
    #enable bash completion in interactive shells  
    #if ! shopt -oq posix; then  
    #  if [-f  /usr/share/bash-completion/bash_completion ]; then  
    #  . /usr/share/bash-completion/bash_completion  
    #  elif [ -f /etc/bash_completion]; then  
    #   . /etc/bash_completion  
    #  fi  
    #fi  

将注释符号#去掉。
最后 source一下 /etc/bash.bashrc即可， 即


    sudo source /etc/bash.bashrc 


参考
https://blog.csdn.net/xxl12345/article/details/76407651

## 1.2解决Ubuntu更新源慢的问题
首先备份源列表:
    sudo cp /etc/apt/sources.list /etc/apt/sources.list_backup

而后用gedit或其他编辑器打开:
    sudo gedit /etc/apt/sources.list

从下面列表中选择合适的源（对应系统版本），替换掉文件中所有的内容，保存编辑好的文件:
清华大学源


    # 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
    
    # 预发布软件源，不建议启用
    # deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
    # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse

阿里云源


    # deb cdrom:[Ubuntu 16.04 LTS _Xenial Xerus_ - Release amd64 (20160420.1)]/ xenial main restricted
    deb-src http://archive.ubuntu.com/ubuntu xenial main restricted #Added by software-properties
    deb http://mirrors.aliyun.com/ubuntu/ xenial main restricted
    deb-src http://mirrors.aliyun.com/ubuntu/ xenial main restricted multiverse universe #Added by software-properties
    deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted
    deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted multiverse universe #Added by software-properties
    deb http://mirrors.aliyun.com/ubuntu/ xenial universe
    deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
    deb http://mirrors.aliyun.com/ubuntu/ xenial multiverse
    deb http://mirrors.aliyun.com/ubuntu/ xenial-updates multiverse
    deb http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse #Added by software-properties
    deb http://archive.canonical.com/ubuntu xenial partner
    deb-src http://archive.canonical.com/ubuntu xenial partner
    deb http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted
    deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted multiverse universe #Added by software-properties
    deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe
    deb http://mirrors.aliyun.com/ubuntu/ xenial-security multiverse



然后，刷新列表:
    sudo apt-get update
参考 https://www.cnblogs.com/webnote/p/5767853.html
清华镜像 https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
阿里镜像源 https://www.cnblogs.com/graye/p/6862939.html

# 2.实现远程协助


    #安装xrdp 
     sudo apt-get install xrdp 
     #安装vnc4server 
     sudo apt-get install vnc4server tightvncserver
     #安装xubuntu-desktop 
     sudo apt-get install xubuntu-desktop 
     #向xsession中写入xfce4-session 
     echo “xfce4-session” >~/.xsession 
     sudo sed -i.bak '/fi/a #xrdp multi-users \n unity \n' /etc/xrdp/startwm.sh
     #开启xrdp服务 
     sudo service xrdp restart

使用WinSCP软件在windows和ubuntu中进行文件传输


    apt-get install ssh
    ps -e |grep ssh


参考 https://www.cnblogs.com/xuliangxing/p/7560723.html
https://blog.csdn.net/haojiahuo50401/article/details/6280681
## 2.1Windows系统远程进入Ubuntu系统时报错：内部错误
解决办法：在  /etc/xrdp/sesman.ini文件的末尾添加以下两行：
    param8 =-SecurityTypes
    param9=None

然后重启xrdp服务：
    sudo /etc/init.d/xrdp restart

参考 https://blog.csdn.net/weixin_39699494/article/details/88553245
## 2.2远程桌面连接XRDP黑屏、鼠标变X解决办法
`gedit /etc/xrdp/startwm.sh` 
#在./etc/X11/Xsession前插入 xfce4-session 
#重启xrdp 
    cd /etc/init.d/ 
    ./xrdp restart

参考 http://blog.sina.com.cn/s/blog_70ad1b620102vtny.html


# 3安装MATLAB
Ubuntu版本MATLAB的安装和破解参考 
https://blog.csdn.net/weixin_41038644/article/details/84680646
创建桌面快捷方式matlab.desktop

    
    [Desktop Entry]
    Type=Application
    Name=Matlab
    GenericName=Matlab 2018b
    Comment=Matlab:The Language of Technical Computing
    Exec=sh /usr/local/MATLAB/R2018b/bin/matlab -desktop
    Icon=/usr/local/MATLAB/R2018b/toolbox/nnet/nnresource/icons/matlab.png
    StartupNotify=true
    Terminal=false
    Categories=Development;Matlab;

解决Untrusted application launcher，通常原因是*.desktop没有执行权限。


    chmod +x *.desktop 

