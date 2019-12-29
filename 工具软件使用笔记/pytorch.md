# 一、PyTorch 是什么

PyTorch是一个基于Torch的Python开源机器学习库，用于自然语言处理等应用程序。它主要由Facebookd的人工智能小组开发，不仅能够 实现强大的GPU加速，同时还支持动态神经网络，这一点是现在很多主流框架如TensorFlow都不支持的。 

# 二、基本语法
介绍pytorch基本知识，对pytorch有基本的了解，构建基本的神经网络。主要参考pytorch官方教程。
## 2.1 张量（Tensors）
张量张量类似于numpy的ndarrays，不同之处在于张量可以使用GPU来加快计算。
对tensor的操作可分为两类：
（1）torch.function，如torch.save等。
（2）tensor.function，如a.view等。
**说明**：函数名以_结尾的都是inplace方式，即会修改调用者自己的数据，如a.add（b），加法的结果仍存储在a中，a被修改了。

### 数据类型变化

为了方便测试，我们构建一个新的张量，你要转变成不同的类型只需要根据自己的需求选择即可

```
tensor = torch.Tensor(3, 5)
```

##### torch.long() 将tensor投射为long类型

```
newtensor = tensor.long()
```

##### torch.half()将tensor投射为半精度浮点类型

```
newtensor = tensor.half()
```

##### torch.int()将该tensor投射为int类型

```
newtensor = tensor.int()
```

##### torch.double()将该tensor投射为double类型

```
newtensor = tensor.double()
```



### Tensor到元素值的转换

x.item()

```python
x = torch.randn(1)
print(x)
print(x.item())

#结果是
tensor([-0.4464])
-0.44643348455429077
```






### Tensor 和 numpy 的转换

```
a=torch.tensor([12.0,11],requires_grad=True)
b=b.data.numpy()
```

我们很容易用`numpy()`和`from_numpy()`将`Tensor`和NumPy中的数组相互转换。但是需要注意的一点是：
**这两个函数所产生的的`Tensor`和NumPy中的数组共享相同的内存（所以他们之间的转换很快），改变其中一个时另一个也会改变！！！**

> 还有一个常用的将NumPy中的array转换成`Tensor`的方法就是`torch.tensor()`, 需要注意的是，此方法总是会进行数据拷贝（就会消耗更多的时间和空间），所以返回的`Tensor`和原来的数据不再共享内存。



### `Tensor` on GPU



用方法`to()`可以将`Tensor`在CPU和GPU（需要硬件支持）之间相互移动。

``` python
# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)                       # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型
```

判读GPU是否可用

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

我们在进行转换时，需要把数据，网络，与损失函数转换到GPU上

1.构建网络时，把网络，与损失函数转换到GPU上

```python
net = LeNet()
net = net.to(device)
print("training on ", device)
print(net)
```

2.训练网络时，把数据转换到GPU上

```python
x = x.to(device)
y = y.to(device)
y_hat = net(x)
l = loss(y_hat, y)
```

3.取出数据是，需要从GPU准换到CPU上进行操作

```python
loss = loss.cpu()
acc = acc.cpu()
train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
```

​		

### 创建Tensor
|函数|说明  |
|:-:|:-:|
| torch.tensor(data ) | 赋值 |
|torch.arange(start,end,step=1,out=None)||
|torch.zeros(*sizes )|全部生成为0 |
|torch.clone()||
|torch.ones(*sizes )|全部生成为1 |
|torch.eye(n,m=None,out=None)|返回一个2维张量，对角线位置全为1，其他位置全0|
|torch.from_numpy(ndarray)|tensor和numpy的转换|
|torch.linspace(start, end, steps=100, out=None) |返回一个1维张量，包含在区间start和end上均匀间隔的step个点。|
|torch.logspace(start,end,steps=100.out=None)|设置的区间为常用对数，输出的值为其对应的真数|

### 创建随机Tensor
|函数|说明  |
|--|--|
|torch.rand(*sizes, out=None)   | 从区间[0, 1)的均匀分布中抽取的一组随机数 |
|torch.randperm(n,out=None)|返回一个从0到n-1的随机整数排列|
|torch.randn(*sizes, out=None)|从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数|
|torch.normal(means, std, out=None) |指定均值means和标准差std的离散正态分布中抽取的一组随机数。|包含在区间start和end上均匀间隔的step个点。|
|uniform(from,to)|均匀分布|

### 常用Tensor方法
|函数|说明  |
|--|--|
|torch.view(*shape) |调整tensor的形状,与源tensor共享内存|
|torch.squeeze(input, dim=None, out=None)|删除尺寸1的输入的所有尺寸的张量|
|torch.unsqueeze(input, dim, out=None) |插入在指定位置的尺寸标注尺寸的新张量。|
|torch.resize|修改tensor的尺寸|
|torch.None|添加一个轴|
|a > 1|返回一个bool矩阵|
|torch.gather(input, dim, index)|根据index，在dim维度上选取数据，输出的size与index一样|
|torch.index_select(input, dim, index)|在指定维度dim上选取，比如选取某些行、某些列|
|torch.masked_select(input, mask)|例子如上，a[a>0]，使用ByteTensor进行选取|
|torch.non_zero(input)|非0元素的下标|

#### advanced indexing

​		通过利用PyTorch  advanced indexing 的属性，你可以用一个二元张量来索引数据张量。这个张量本质上是把数据过滤成索引张量中只对应于1的项(或行)。  

```python
bad_indexes = torch.le(target, 3)
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum()
# Out[13]:
(torch.Size([4898]), torch.uint8, tensor(20))
# In[14]:
bad_data = data[bad_indexes]
bad_data.shape
# Out[14]:
torch.Size([20, 11])
```



### 数值计算

|函数|说明  |
|--|--|
|abs/sqrt/div/exp/fmod/log/pow..|绝对值/平方根/除法/指数/求余/求幂..|
|cos/sin/asin/atan2/cosh..|相关三角函数|
|ceil/round/floor/trunc| 上取整/四舍五入/下取整/只保留整数部分|
|clamp(input, min, max)|超过min和max部分截断|
|sigmod/tanh..|激活函数|
| torch.lerp(star, end, weight)  |  返回结果是out= star t+ (end-start) * weight |
|  torch.equal(torch.Tensor(a), torch.Tensor(b))  | 两个张量进行比较，如果相等返回true，否则返回false|
|torch.max(input)| 返回输入元素的最大值|
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
|addmm/addbmm/addmv/addr/badbmm..|矩阵运算|
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

torch.Tensor 是包的核心类。如果将其属性 .requires_grad 设置为 True，则会开始跟踪针对 tensor 的所有操作。完成计算后，您可以调用 .backward() 来自动计算所有梯度。该张量的梯度将累积到 .grad 属性中。

``` pythoncript
x = torch.ones(2, 2, requires_grad=True)
a = torch.randn(2, 2)
a.requires_grad_(True)
print(a.requires_grad)

```

> 注意在`y.backward()`时，如果`y`是标量，则不需要为`backward()`传入任何参数；否则，需要传入一个与`y`同形的`Tensor`。为什么在`y.backward()`时，如果`y`是标量，则不需要为`backward()`传入任何参数；否则，需要传入一个与`y`同形的`Tensor`?
> 简单来说就是为了避免向量（甚至更高维张量）对张量求导，而转换成标量对张量求导。举个例子，假设形状为 `m x n` 的矩阵 X 经过运算得到了 `p x q` 的矩阵 Y，Y 又经过运算得到了 `s x t` 的矩阵 Z。那么按照前面讲的规则，dZ/dY 应该是一个 `s x t x p x q` 四维张量，dY/dX 是一个 `p x q x m x n`的四维张量。问题来了，怎样反向传播？怎样将两个四维张量相乘？？？这要怎么乘？？？就算能解决两个四维张量怎么乘的问题，四维和三维的张量又怎么乘？导数的导数又怎么求，这一连串的问题，感觉要疯掉…… 
> 为了避免这个问题，我们**不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量**。所以必要时我们要把张量通过将所有张量的元素加权求和的方式转换为标量，举个例子，假设`y`由自变量`x`计算而来，`w`是和`y`同形的张量，则`y.backward(w)`的含义是：先计算`l = torch.sum(y * w)`，则`l`是个标量，然后求`l`对自变量`x`的导数。

如果不想要被继续追踪，可以调用`.detach()`将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了。此外，还可以用`with torch.no_grad()`将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，我们并不需要计算可训练参数（`requires_grad=True`）的梯度。

`		Function`是另外一个很重要的类。`Tensor`和`Function`互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）。每个`Tensor`都有一个`.grad_fn`属性，该属性即创建该`Tensor`的`Function`, 就是说该`Tensor`是不是通过某些运算得到的，若是，则`grad_fn`返回一个与这些运算相关的对象，否则是None。

下面通过一些例子来理解这些概念。  


## 2.3 构建神经网络
​	autograd实现了自动微分系统，然而对于深度学习来说过于底层，其抽象程度较低，如果用其来实现深度学习模型，则需要编写的代码量极大。在这种情况下，torch.nn应运而生，其是专门为深度学习设计的模块。torch.nn的核心数据结构是Module，它是一个抽象的概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。

1.定义一个包含可训练参数的神经网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
params = list(net.parameters())
```

2.迭代整个输入

```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
```

3.通过神经网络处理输入

4.计算损失(loss)

```
criterion = nn.MSELoss()
loss = criterion(output, target)
```

5.反向传播梯度到神经网络的参数

```
net.zero_grad()
out.backward()
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
#如果你是用神经网络，你想使用不同的更新规则，类似于 SGD, Nesterov-SGD, Adam, RMSProp, 等。为了让这可行，我们建立了一个小包：torch.optim 实现了所有的方法。使用它非常的简单。
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```



## 2.4 并行化

​	如何用 DataParallel 来使用多 GPU。

```python
#通过 PyTorch 使用多个 GPU 非常简单。将模型放在一个 GPU：
 device = torch.device("cuda:0")
 model.to(device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#可以复制所有的张量到 GPU：
mytensor = my_tensor.to(device)
```

​	尽管如此，PyTorch 默认只会使用一个 GPU。通过使用 DataParallel 让你的模型并行运行，你可以很容易的在多 GPU 上运行你的操作。

```python
model = nn.DataParallel(model)
```

​	

# 三、常用函数

​	






### 索引，切片，连接，换位
|函数|说明  |
|--|--|
|torch.cat(inputs,dimension=0)|在给定维度上对输入的张量序列进行连接操作|
|torch.squeeze(inout,dim=None,out=None)|将输入张量形状中的1去除|
|torch.expand(*sizes) |单个维度扩大为更大的尺寸。|
|index_add_(dim, index, tensor) |按参数index中的索引数确定的顺序，将参数tensor中的元素加到原来的tensor中。|
|repeat(*sizes) |沿着指定的维度重复tensor。|
|torch.reshape(input, shape) |
|torch.transpose(input, dim0, dim1)|转置|
|torch.masked_select(input, mask, out=None) |根据二进制掩码对输入进行索引，这是一个新的索引|
|torch.chunk(tensor, chunks, dim=0) |将张量拆分为特定数量的“块”|
|torch.narrow(input, dimension, start, length)|
|torch.stack(seq, dim=0, out=None) |拼接|

### 存储和提取模型参数

```python
torch.save(net1, 'net.pkl')  # save entire net
torch.save(net1.state_dict(), 'net_params.pkl')   # save only the parameters
net3.load_state_dict(torch.load('net_params.pkl'))

# 存储Tensor
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
```

### 批量的数据

```
import torch.utils.data as Data
BATCH_SIZE = 5
x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)
```



### 常见优化器

```
opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
```



### 定义一个卷积层

```
super(CNN, self).__init__()
self.conv1 = nn.Sequential(         # input shape (1, 28, 28)    
nn.Conv2d(        
in_channels=1,              # input height        
out_channels=16,            # n_filters        
kernel_size=5,              # filter size        
stride=1,                   # filter movement/step        
padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1    
),                              # output shape (16, 28, 28)    
nn.ReLU(),                      # activation
nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
)
```





### 定义一个RNN

```
super(RNN, self).__init__()
self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns    
input_size=INPUT_SIZE,    
hidden_size=64,         # rnn hidden unit    
num_layers=1,           # number of rnn layer    
batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
)
```



### onehot

```
batch_size = 2
sequence_len = 3
hidden_dim = 5
x = torch.zeros(batch_size, sequence_len, hidden_dim).scatter_(dim=-1,
                               index=torch.LongTensor([[[2],[2],[1]],[[1],[2],[3]]]),
                               value=1)

print(x)
————————————————
版权声明：本文为CSDN博主「guotong1988」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/guotong1988/article/details/102541546
```



参考文献：
[pytorch官网](https://pytorch.org/)
深度学习框架pytorch：入门与实践
[PyTorch 深度学习:60分钟快速入门](https://blog.csdn.net/u014630987/article/details/78669051)
[torch---pytorch常用函数](https://blog.csdn.net/qq_24407657/article/details/81835614)
[Pytorch常用函数操作总结](https://blog.csdn.net/qq_35012749/article/details/88235837)

[Pytorch官网教程中文版](http://pytorch123.com/SecondSection/neural_networks/)