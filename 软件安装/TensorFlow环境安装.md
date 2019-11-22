# Python 3.6

    yum install epel-release -y
    yum install https://centos7.iuscommunity.org/ius-release.rpm -y
    yum install python36u -y
    yum install python36u-devel -y
    ln -s /bin/python3.6 /bin/python3
    yum install python36u-pip -y
    ln -s /bin/pip3.6 /bin/pip3
参考 
https://blog.51cto.com/wenguonideshou/2083301

首先安装 Python 3.6，这里使用 Anaconda 3 来安装

    wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
下载完成之后，在 anaconda 所在目录的终端输入：
```
bash Anaconda3-5.2.0-Linux-x86_64.sh
```
指定了将其安装到 /usr/local/anaconda3 目录下，全局安装，所有用户共享。
选择把 /usr/local/anaconda3/bin 目录添加到环境变量中，添加如下内容：

```
sudo gedit /etc/profile
export PATH=/usr/local/anaconda3/bin${PATH:+:${PATH}}
```
最后可以执行：验证下 python3、pip 命令是否都来自 Anaconda

```
pip -V
which python3
python
```

参考
https://cloud.tencent.com/developer/article/1086781
https://blog.csdn.net/moses1994/article/details/81507802
# 1.安装Nvidia驱动，用 apt-get 安装
添加开源显驱仓库

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
```
安装一些依赖
```
sudo apt-get install dkms synaptic build-essential
```
进入命令界面后，更改系统黑名单中的禁用名单。
```
sudo chmod 666 /etc/modprobe.d/blacklist.conf
sudo vim /etc/modprobe.d/blacklist.conf
```
添加以下黑名单列表在blacklist.conf文本后面

```
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist rivatv
blacklist nvidiafb
```
更新内核,并重启

```
sudo update-initramfs -u
sudo reboot
```

同样在命令行中对禁用情况进行核实

```
lsmod | grep nouveau
```
然后，下载NVIDIA 驱动 
下载地址 https://www.nvidia.com/Download/index.aspx?lang=en-us

重启Ubuntu系统，使用ctr+atl+F1进入终端界面。
关闭图形化界面

```
sudo service lightdm stop
```
安装驱动

```
sudo chmod u+x NVIDIA.run
sudo ./NVIDIA.run
```
最后重启图形化界面
```
sudo service lightdm start
```
查看是否安装成功  

```
nvidia-smi
```

然后寻找最佳的驱动

```
sudo ubuntu-drivers devices
```
安装推荐的 nvidia-*第三方驱动

```
sudo apt-get install nvidi-396
```
然后安装好后重启电脑就可以了

检查是否安装到位

```
sudo nvidia-smi

sudo nvidia-settings
```

参考 
https://www.jianshu.com/p/0e5a18d8dadb 不踩坑：在Ubuntu下安装TensorFlow的最简单方法（无需手动安装CUDA和cuDNN）
https://www.jianshu.com/p/4763b23aafb9

# 2. 安装cuda9.0+cudnn7.3+tensorflow-gpu-1.10
tensorflow gpu   cuda 关系
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190612162922759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXN0bnV0c3M=,size_16,color_FFFFFF,t_70)
https://www.tensorflow.org/install/source#common_installation_problems
## cuda9.0安装
首先去英伟达官网下载cuda安装包：
https://developer.nvidia.com/cuda-toolkit-archive
安装cuda过程中，注意**第二部步n（不安装driver）**

然后我们需要在/etc/profile，加上下面两行命令，这是为了让所有用户都能获取环境变量的配置：
```
export PATH=/usr/local/cuda-9.0/bin/${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
由于更改 profile 文件以后不会立即生效，此时要么重启终端，要么在命令行执行上面两条指令。
至此， cuda9.0 安装完毕。 下面测试一下 cuda 是否成功安装。

参考
https://blog.csdn.net/weixin_42452024/article/details/87993246
https://blog.csdn.net/wanzhen4330/article/details/81699769

## cudnn7安装
下载CUDNN7并安装,这里只需要下载下图所示三个包
https://developer.nvidia.com/rdp/cudnn-download 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190611152414102.png)	

```
sudo dpkg -i libcudnn7_7.3.0.29-1+cuda9.0_amd64.deb
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019061115262279.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXN0bnV0c3M=,size_16,color_FFFFFF,t_70)
在执行第4条即./mnistCUDNN命令，出现：
/mnistCUDNN: error while loading shared libraries: libcudart.so.9.0: cannot open shared object file: No such file or directory
解决方法

```
sudo cp /usr/local/cuda-9.0/lib64/libcudart.so.9.0 /usr/local/lib/libcudart.so.9.0 && sudo ldconfig
sudo cp /usr/local/cuda-9.0/lib64/libcublas.so.9.0 /usr/local/lib/libcublas.so.9.0 && sudo ldconfig
sudo cp /usr/local/cuda-9.0/lib64/libcurand.so.9.0 /usr/local/lib/libcurand.so.9.0 && sudo ldconfig
```


参考
https://blog.csdn.net/lengconglin/article/details/77506386
https://blog.csdn.net/lingyunxianhe/article/details/85012003



## tensorflow-gpu-1.10安装
增加第三方镜像连接

```
conda config --add channels r # R软件包
conda config --add channels conda-forge # Conda社区维护的不在默认通道中的软件
```
【删除第三方链接，恢复到默认设置】

```
conda config --remove-key channels
conda config --show # 查看conda的配置，确认channels
```


激活tensorflow环境

```
conda create python=3.6 --prefix=/software/anaconda3/envs/tensorflow
```
pip安装tensorflow-gpu，并使用豆瓣镜像加速下载

```
pip install tensorflow-gpu==1.10 -i https://pypi.doubanio.com/simple/
```

如果想让Ubuntu下所有用户都能够使用anaconda，则需要将anaconda安装路径添加到系统变量中：

```
gedit /etc/profile
```


验证GPU是否调用成功

```
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
a = tf.constant([1.,2.,3.,4.,5.,6.], shape=[2,3], name='a')
b = tf.constant([1.,2.,3.,4.,5.,6.], shape=[3,2], name='b')
c = tf.matmul(a,b)

with tf.Session(config= tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
```
Tensorflow：ImportError：libcusolver.so.8.0：无法打开共享对象文件

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
sudo apt install nvidia-cuda-dev
```

参考
https://zhuanlan.zhihu.com/p/64766956
https://baijiahao.baidu.com/s?id=1604501192403223852&wfr=spider&for=pc 小叮当机器学习：Python3.6配置TensorFlow的GPU版详细安装教程
https://blog.csdn.net/u013082989/article/details/83382230
https://vimsky.com/article/3597.html Tensorflow：ImportError：libcusolver.so.8.0：无法打开共享对象文件：没有这样的文件或目录
https://vimsky.com/article/3597.html

https://blog.csdn.net/qq_30683995/article/details/82619859

## 安装pytorch
在[pytorch官网](https://pytorch.org/)根据配置，生成pip命令：

    pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190915201043272.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXN0bnV0c3M=,size_16,color_FFFFFF,t_70)


## 安装pycharm
下载pycharm社区版本，解压即可使用
破解版的安装参考 https://www.jianshu.com/p/8abc6e0ddb7e

创建pycharm快捷方式
新建文件pycharm.desktop

```
[Desktop Entry]
Type=Application
Name=Pycharm
GenericName=Pycharm3
Comment=Pycharm3:The Python IDE
Exec="/usr/local/pycharm-community-2019.1.3//bin/pycharm.sh" %f
Icon=/usr/local/pycharm-community-2019.1.3/bin/pycharm.png
Terminal=pycharm
Categories=Pycharm;
```
添加anaconda的python编译器 
查找python编译器的位置命令：
which python 
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019061220104489.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190612200925190.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXN0bnV0c3M=,size_16,color_FFFFFF,t_70)
LD_LIBRARY_PATH
pycharm远程调试ImportError:libcusolver.so.8.0: cannot open shared object file: No such file or directory
将PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 添加到默认环境变量中
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019061310565492.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXN0bnV0c3M=,size_16,color_FFFFFF,t_70)
参考
https://blog.csdn.net/qq_34654240/article/details/80480849
