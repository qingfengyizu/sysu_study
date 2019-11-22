# centos7系统安装
centos7系统安装
https://www.jb51.net/os/128751.html
**启动网络**
输入 ifconfig  查看网卡（默认第一个）
在终端输入 gedit /etc/sysconfig/network-script/ifcfg-ens33 回车。（**ifcfg-ens33为网卡名称**）进入后找到 **ONBOOT** 项，将no改为yes保存退出。
最后 systemctl restart network.service   重启网卡即可。
 参考 https://jingyan.baidu.com/article/9c69d48ff9143b13c9024e04.html
 
**安装ssh**
安装openssh-server

    yum install -y openssl openssh-server

 修改配置文件

     gedit  /etc/ssh/sshd_config

将 PermitRootLogin，RSAAuthentication，PubkeyAuthentication的设置yes
启动ssh的服务：

    systemctl start sshd.service

设置开机自动启动ssh服务

    systemctl enable sshd.service

参考 https://my.oschina.net/laiconglin/blog/675317

**nvidia驱动安装**
首先都要安装kernel

     yum -y install gcc kernel-devel kernel-headers
     rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
     rpm -Uvh http://www.elrepo.org/elrepo-release-7.0-2.el7.elrepo.noarch.rpm
     yum install yum-plugin-fastestmirror
屏蔽默认带有的nouveau
打开  /lib/modprobe.d/dist-blacklist.conf
将nvidiafb注释掉。

    #blacklist nvidiafb
然后添加以下语句：

```

blacklist nouveau
options nouveau modeset=0
```

重建initramfs image（强烈建议复制）

```
mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak
dracut /boot/initramfs-$(uname -r).img $(uname -r)
```
修改运行级别为文本模式

```
systemctl set-default multi-user.target
```
重新启动, 使用root用户登陆

进入下载的驱动所在目录，安装驱动
修改运行级别回图形模式

```
systemctl set-default graphical.target
```
 

参考 https://blog.csdn.net/u013378306/article/details/69229919

**xmanager远程连接**
yum install epel-release
yum install lightdm
vim /etc/lightdm/lightdm.conf

    [XDMCPServer]
    enabled=true
    port=177

systemctl disable gdm
systemctl enable lightdm
systemctl start lightdm
yum groupinstall xfce
systemctl stop firewalld.service

解决在Xmanager下启动的应用程序键盘输入两次的问题
https://blog.csdn.net/uplife1980/article/details/84574055
解决Xmanager连接CentOS7远程桌面 报错Failed to connect to socket /tmp/dbus-xxxxxxx: Connection refused
https://blog.csdn.net/one312/article/details/81075795
参考 
https://www.cnblogs.com/zuxing/articles/8986494.html
https://www.jianshu.com/p/eea260629b19
https://www.wandouip.com/t5i331184/

**安装TensorFlow环境**
参考 https://www.jianshu.com/p/78a936c27ec4

**安装teamview**

qt5-qtwebkit下载地址
https://centos.pkgs.org/7/epel-x86_64/qt5-qtwebkit-5.9.1-1.el7.x86_64.rpm.html 

    yum install qt5-qtwebkit-5.9.1-1.el7.x86_64.rpm
    yum -y install wget
    wget https://download.teamviewer.com/download/linux/teamviewer.x86_64.rpm
    yum -y install teamviewer.x86_64.rpm

参考 
https://www.hellojava.com/a/45012.html
https://www.itzgeek.com/how-tos/linux/centos-how-tos/how-to-install-teamviewer-on-centos-7-rhel-7.html

**Centos7 ifconfig这个命令没找到的解决方法**
1.查看ifconfig 命令存在情况
这时我们可以使用命令查看一下它是否存在，ls /sbin/ifconfig。
能看到存在的话我们在继续执行命令echo $PATH，这样我们看这个命令是不是包括在环境变量里面，如果没有使用export PATH=$PATH:/usr/sbin添加到当前环境变量中。
2.ifconfig 命令不存在
yum -y install net-tools
等待安装完毕之后我们就可以直接使用
参考
[Centos7 ifconfig这个命令没找到的解决方法（转）](https://www.cnblogs.com/yangfeilong/p/9797933.html)

### Centos 7 Linux系统修改网卡名称为ethx
1、编辑网卡信息
2、修改grub
3、重启验证是否修改成功
参考
[Centos 7 Linux系统修改网卡名称为ethx](https://www.cnblogs.com/Wolf-Dreams/p/9090577.html)
[centos7修改网卡名称为eth0](https://www.cnblogs.com/freeblogs/p/7881597.html)