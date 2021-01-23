# 一、前期工作
## 软件和工具下载
vcs_2016

verdi_2016

PT_2016

DC_2016

scl_v11.9

synopsys_installer

scl_keygen

[\[资料\] DesignCompiler2016和PrimeTime2016的软件破解与安装](http://bbs.eetop.cn/thread-763341-1-1.html)
[从零开始VCS+Verdi 安装过程](https://blog.csdn.net/Ztrans/article/details/88757695)

# 二、软件安装
## 2.1 安装synopsys installer
运行SynopsysInstaller_v3.3.run，终端./SynopsysInstaller_v3.3.run，输入安装路径，将synopsys installer安装到文件夹，得到~/synopsys/install中的setup.sh文件。
## 2.2 安装EDA软件
`./setup.sh` 进入图形化安装界面。
# 三、后续安装问题
### 3.1获取mac地址
	ifconfig eth0 |egrep "ether" 

###  3.2修改Linux系统修改网卡名称为eth0
1、编辑网卡信息
2、修改grub
3、重启验证是否修改成功

参考
[Centos 7 Linux系统修改网卡名称为ethx](https://www.cnblogs.com/Wolf-Dreams/p/9090577.html)
[centos7修改网卡名称为eth0](https://www.cnblogs.com/freeblogs/p/7881597.html)

### 3.3启动license安装lsb
	yum install redhat-lsb
查看或者清空27000端口
netstat -ap | grep 27000
kill -9 XXX(看到的占用端口的ID号)

### 3.4获取license
参考 [VCS+Verdi 安装及破解过程（CentOS7)-----FPGA开发](https://blog.csdn.net/qq_40829605/article/details/85345795) [最新版本](http://bbs.eetop.cn/thread-877536-1-1.html)


### 3.4 添加环境变量


    #scl
    export PATH=$PATH:/home/synopsys/SCL-11.9/amd64/bin/
    
    #LICENCE
    export LM_LICENSE_FILE=27000@localhost.localdomain
    alias license_synopsys='/home/synopsys/SCL-11.9/amd64/bin/lmgrd -c /home/synopsys/SCL-11.9/Synopsys.dat'
    
    #DC
    export PATH=$PATH:/home/synopsys/DC-L-2016.03-SP1/bin/
    export PATH=$PATH:/home/synopsys/DC-L-2016.03-SP1/linux64/
    alias dc='design_vision'
    
    #ICC
    export PATH=$PATH:/home/synopsys/ICC-L-2016.03-SP1/bin/
    export PATH=$PATH:/home/synopsys/ICC-L-2016.03-SP1/linux64/
    alias icc='icc_shell -gui'


​    
    #VCS
    export DVE_HOME=/home/synopsys/VCS-L-2016.06/gui/dve
    export PATH=$PATH:/home/synopsys/VCS-L-2016.06/gui/dve/bin
    export VCS_ARCH_OVERRIDE=linux
    export VCS_HOME=/home/synopsys/VCS-L-2016.06
    export PATH=$PATH:/home/synopsys/VCS-L-2016.06/bin
    alias dve='dve -full64'
    alias vcs='vcs'
    
    #VERDI
    export PATH=$PATH:/home/synopsys/Verdi3_L-2016.06-1/
    export VERDI_HOME=/home/synopsys/Verdi3_L-2016.06-1/
    export PATH=$PATH:/home/synopsys/Verdi3_L-2016.06-1/bin/
    alias verdi='verdi -full64'
    
    #PT
    export PATH=$PATH:/home/synopsys/PT-M-2016.12-SP1/bin/
    export PATH=$PATH:/home/synopsys/PT-M-2016.12-SP1/linux64/
    alias pt='pt_shell -gui'
    #alias pt='pt_shell -multi_scenario'


### 3.6 添加缺损库
#### verdi缺少libXss.so.1库

    sudo yum install libXScrnSaver

#### pt缺少libtiff.so.3库
首先根据提示，执行`sudo yum -y install libtiff.so.3`安装函数库，但安装完后发现还是提示刚才的错误，进入/usr/lib，发现这个函数库确实已经安装好了。
这时我们进入/usr/lib64目录下，并执行`ln -s libtiff.so.5 libtiff.so.3`，然后问题解决了。
参考：[装Custom Waveview时error while loading shared libraries: libtiff.so.3](http://bbs.eetop.cn/thread-762453-1-1.html)

## 3.7 修改hostname 

```
gedit etc/hosts
最后添加 127.0.0.1 pcname 
```

参考 [博客](http://www.360doc.com/content/14/1127/17/14187254_428530562.shtml)和[博客](http://bbs.eetop.cn/thread-306118-1-1.html)

# 说明！！仅做学习使用！！无商业用途！！
参考：
