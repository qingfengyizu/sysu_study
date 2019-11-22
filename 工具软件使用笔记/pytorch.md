# 一、PyTorch 是什么

PyTorch是一个基于Python的科学计算包，相比numpy能使用GPU来加快计算。

# 二、基本语法
介绍pytorch基本知识，对pytorch有基本的了解，构建基本的神经网络。主要参考pytorch官方教程。
## 2.1 张量（Tensors）
张量张量类似于numpy的ndarrays，不同之处在于张量可以使用GPU来加快计算。
对tensor的操作可分为两类：
（1）torch.function，如torch.save等。
（2）tensor.function，如a.view等。
**说明**：函数名以_结尾的都是inplace方式，即会修改调用者自己的数据，如a.add（b），加法的结果仍存储在a中，a被修改了。

### 创建Tensor
|函数|说明  |
|--|--|
| torch.tensor(data ) | 赋值 |
|torch.arange(start,end,step=1,out=None)|
|torch.zeros(*sizes )|全部生成为0 
|torch.clone()||
|torch.ones(*sizes )|全部生成为1 |
|torch.eye(n,m=None,out=None)|返回一个2维张量，对角线位置全为1，其他位置全0
|torch.from_numpy(ndarray)|tensor和numpy的转换|
|torch.linspace(start, end, steps=100, out=None) |返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
|torch.logspace(start,end,steps=100.out=None)|设置的区间为常用对数，输出的值为其对应的真数

### 创建随机Tensor
|函数|说明  |
|--|--|
|torch.rand(*sizes, out=None)   | 从区间[0, 1)的均匀分布中抽取的一组随机数 |
|torch.randperm(n,out=None)|返回一个从0到n-1的随机整数排列|
|torch.randn(*sizes, out=None)|从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数|
|torch.normal(means, std, out=None) |指定均值means和标准差std的离散正态分布中抽取的一组随机数。|包含在区间start和end上均匀间隔的step个点。|
|uniform(from,to)|均匀分布

### 常用Tensor操作
|函数|说明  |
|--|--|
|view(*shape) |调整tensor的形状,与源tensor共享内存|
|torch.squeeze(input, dim=None, out=None)|删除尺寸1的输入的所有尺寸的张量|
|torch.unsqueeze(input, dim, out=None) |插入在指定位置的尺寸标注尺寸的新张量。|
|resize|修改tensor的尺寸|
|None|添加一个轴|
|a > 1|返回一个bool矩阵|
gather(input, dim, index)|根据index，在dim维度上选取数据，输出的size与index一样
index_select(input, dim, index)|在指定维度dim上选取，比如选取某些行、某些列
masked_select(input, mask)|例子如上，a[a>0]，使用ByteTensor进行选取
non_zero(input)|非0元素的下标

### 数值计算
|函数|说明  |
|--|--|
|abs/sqrt/div/exp/fmod/log/pow..|绝对值/平方根/除法/指数/求余/求幂..|
|cos/sin/asin/atan2/cosh..|相关三角函数|
|ceil/round/floor/trunc| 上取整/四舍五入/下取整/只保留整数部分|
|clamp(input, min, max)|超过min和max部分截断|
|sigmod/tanh..|激活函数
| torch.lerp(star, end, weight)  |  返回结果是out= star t+ (end-start) * weight |
 |  torch.equal(torch.Tensor(a), torch.Tensor(b))  | 两个张量进行比较，如果相等返回true，否则返回false
 |torch.max(input)| 返回输入元素的最大值
 |mean/sum/median/mode|均值/和/中位数/众数|
|norm/dist|范数/距离|
|std/var|标准差/方差|
|cumsum/cumprod|累加/累乘|
|gt/lt/ge/le/eq/ne|大于/小于/大于等于/小于等于/等于/不等|
|topk|最大的k个数|
|sort|排序|
|max/min|比较两个tensor最大最小值|
### 线性代数计算
|函数|说明  |
|--|--|
|trace|对角线元素之和(矩阵的迹)|
|diag|对角线元素|
|triu/tril|矩阵的上三角/下三角，可指定偏移量|
|mm/bmm|矩阵乘法，batch的矩阵乘法|
|addmm/addbmm/addmv/addr/badbmm..|矩阵运算
|t|转置|
|dot/cross|内积/外积
|inverse|求逆矩阵
|svd|奇异值分解
## 2.2 Autograd: 自动求导(automatic differentiation)
PyTorch在autograd模块中实现了计算图的相关功能，autograd中的核心数据结构是Variable。Variable 封装了tensor，并记录对tensor的操作记录用来构建计算图。Variable的数据结构如图3-4所示，主要包含三个属性。
- data：保存variable所包含的tensor。
- grad：保存data对应的梯度，grad也是variable，而非tensor，它与data形状一致。
- grad_fn：指向一个Function，记录variable的操作历史，即它是什么操作的输出，用来构建计算图。如果某一个变量是由用户创建的，则它为叶子节点，对应的grad_fn等于None。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190916110230663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXN0bnV0c3M=,size_16,color_FFFFFF,t_70)
Variable 支持大部分tensor支持的函数，但其不支持部分inplace函数，因为这些函数会修改tensor自身，而在反向传播中，variable需要缓存原来的tensor来计算梯度。
如果想要计算各个Variable的梯度，只需调用根节点variable的backward方法，autograd会自动沿着计算图反向传播，计算每一个叶子节点的梯度。


用out.backward()来执行反向传播。此Tensor的梯度将累积到.grad属性中。如果不想要被继续追踪，可以调用.detach()将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了。此外，还可以用with torch.no_grad()将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，我们并不需要计算可训练参数（requires_grad=True）的梯度。注意：grad在反向传播过程中是累加的（accumulated），这意味着每次运行反向传播，梯度都会累加之前的梯度，所以反向传播之前需把梯度清零x.grad.data.zero_()。


## 2.3 构建神经网络
autograd实现了自动微分系统，然而对于深度学习来说过于底层，其抽象程度较低，如果用其来实现深度学习模型，则需要编写的代码量极大。在这种情况下，torch.nn应运而生，其是专门为深度学习设计的模块。torch.nn的核心数据结构是Module，它是一个抽象的概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。



# 三、常用函数








## 索引，切片，连接，换位
|函数|说明  |
|--|--|
|torch.cat(inputs,dimension=0)|在给定维度上对输入的张量序列进行连接操作|
|torch.squeeze(inout,dim=None,out=None)|将输入张量形状中的1去除|
|torch.expand(*sizes) |单个维度扩大为更大的尺寸。|
|index_add_(dim, index, tensor) |按参数index中的索引数确定的顺序，将参数tensor中的元素加到原来的tensor中。|
|repeat(*sizes) |沿着指定的维度重复tensor。
|torch.reshape(input, shape) |

|.torch.transpose(input, dim0, dim1)|转置|
|.torch.masked_select(input, mask, out=None) |根据二进制掩码对输入进行索引，这是一个新的索引|
|torch.chunk(tensor, chunks, dim=0) |将张量拆分为特定数量的“块”|
|torch.narrow(input, dimension, start, length)|
|torch.stack(seq, dim=0, out=None) |拼接|



参考文献：
[pytorch官网](https://pytorch.org/)
深度学习框架pytorch：入门与实践
[PyTorch 深度学习:60分钟快速入门](https://blog.csdn.net/u014630987/article/details/78669051)
[torch---pytorch常用函数](https://blog.csdn.net/qq_24407657/article/details/81835614)
[Pytorch常用函数操作总结](https://blog.csdn.net/qq_35012749/article/details/88235837)