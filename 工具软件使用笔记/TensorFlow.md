## 一、TensorFlow是什么
TensorFlow 是由 Google Brain 团队为深度神经网络（DNN）开发的功能强大的开源软件库，加速神经网络代码的实现。
## 二、基本框架
代码分为以下三个主要部分：
- 第一部分 import 模块包含代码将使用的所有库。
- 第二部分 包含图形定义部分,创建想要的计算图。
**计算图：** 是包含节点和边的网络。本节定义所有要使用的数据，也就是张量（tensor）对象（常量、变量和占位符），同时定义要执行的所有计算，即运算操作对象（Operation Object，简称 OP）。每个节点可以有零个或多个输入，但只有一个输出。网络中的节点表示对象（张量和运算操作），边表示运算操作之间流动的张量。
- 第三部分 通过会话执行计算图。
**计算图的执行：** 使用会话对象来实现计算图的执行。会话对象封装了评估张量和操作对象的环境。这里真正实现了运算操作并将信息从网络的一层传递到另外一层。不同张量对象的值仅在会话对象中被初始化、访问和保存。在此之前张量对象只被抽象定义，在会话中才被赋予实际的意义。
### 2.1、定义张量
张量，可理解为一个 n 维矩阵，所有类型的数据，包括标量、矢量和矩阵等都是特殊类型的张量。
TensorFlow 支持以下三种类型的张量：
- **常量**：常量是其值不能改变的张量。
- **变量**：当一个量在会话中的值需要更新时，使用变量来表示。例如，在神经网络中，权重需要在训练期间更新，可以通过将权重声明为变量来实现。变量在使用前需要被显示初始化。
- **占位符**：用于将值输入 TensorFlow 图中。它们可以和 feed_dict 一起使用来输入数据。
### 常量
|函数|说明  |
|--|--|
| tf.constant(4) | 声明一个标量常量  |
|tf.zeros([M,N],tf.dtype)|创建一个所有元素为零的张量|
|tf.zeros_like(t_2)|创建与现有 Numpy 数组或张量常量具有相同形状的张量常量|
|tf.ones([M,N],tf,dtype)|创建一个所有元素都设为 1 的张量。|
|tf.linspace(start,stop,num)|在一定范围内生成一个从初值到终值等差排布的序列|
|tf.range(start,limit,delta)|从开始（默认值=0）生成一个数字序列，增量为 delta（默认值=1），直到终值（但不包括终值）|
|t_random=tf.random_normal([2,3],mean=2.0,stddev=4,seed=12)|创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数组|
|t_random=tf.truncated_normal([1,5],stddev=2,seed=12)|创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的截尾正态分布随机数组|
|t_random=tf.random_uniform([2,3],maxval=4,seed=12)|种子的 [minval（default=0），maxval] 范围内创建形状为 [M，N] 的给定伽马分布随机数组|
|tf.random_shuffle(t_random)|沿着它的第一维随机排列张量|
| tf.convert_to_tensor() |可以将给定的值转换为张量类型|


### 变量
以神经网络中权重和偏置变量举例说明：

    weights=tf.Variable(tf.random_normal([100,100],stddev=2))
    bias=tf.Variable(tf.zeros[100],name='biases')

变量的定义将指定变量如何被初始化，但是必须显式初始化所有的声明变量。在计算图的定义中通过声明初始化操作对象来实现：

    intial_op=tf.global_variables_initializer().
tf.trainable_variables()可以查看所有需要优化的参数。

### 占位符
    tf.placeholder(dtype,shape=None,name=None)

### 2.2 执行会话
方法一：通过Python的上下文管理器来使用会话

    with tf. Session() as sess: 
    	prin(sess. run(v_add))
方法二：明确需要调用会生成函数和关闭会话函数

```
sess=tf.Session()
print(ses.run(tv_add))
sess.close()
```

方法三：交互式环境下直接构建默认会话的方法。

```
sess=tf.InteractiveSession()
print(v_add.eval())
sess.close()
```
### 2.3 优化器
- 梯度下降优化器

```
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step=optimizer.minimize(loss)
```

- Adadelta和RMSprop自适应的、单调递减的学习率

```
optimizer=tf. train. AdadeltaOptimizer(learning_rate=0.8, rho=0.95). minimize(loss)
optimizer=tf. train. RMSpropOptimizer(learning_rate=0.01, decay=0.8, momentum=0.1). minimize(loss)

```
- Adam 优化器。该方法利用梯度的一阶和二阶矩对不同的系数计算不同的自适应学习率

```
optimizer=tf.train.AdamOptimizer().minimize(loss)
```

### 2.4 激活函数
|函数|说明  |
|--|--|
| tf.Sigmoid |  |
|tf.tanh()|双曲正切激活函数|
|tf.relu()|ReLU|
|tf.nn.softmax()|Softmax |





## 三、TensorBoard可视化数据流图
TensorFlow 使用 TensorBoard 来提供计算图形的图形图像。这使得理解、调试和优化复杂的神经网络程序变得很方便。
tf.summary.scalar——对标量数据汇总和记录 
tf.summary.histogram——记录数据的直方图 
tf.summary.merge_all——合并默认图像中的所有汇总
tf.summary.FileWriter——用于将汇总数据写入磁盘
tensorboard --logdir=summary_dir

## 四、常用函数

### Casting
|函数|说明  |
|--|--|
| tf.cast(x, dtype, name=None) |类型的转换  |

### Reduce
|函数|说明  |
|--|--|
|tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None) |在指定维度上求和
|tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None) |在指定维度上求最大值
|tf.reduce_min(input_tensor, reduction_indices=None, keep_dims=False, name=None) |在指定维度上求最小值
|tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)| 在指定维度上求均值


### shape 
|函数|说明  |
|--|--|
|tf.shape(input, name=None) |获取张量的形状
|tf.reshape(tensor, shape, name=None) |变换张量形状
|tf.squeeze(input, squeeze_dims=None, name=None)| 压缩(去除)张量维度（压缩形状大小为1的维度）
|tf.expand_dims(input, dim, name=None) |扩充张量维度

### Slicing and Joining 切片和连接
|函数|说明  |
|--|--|
|tf.slice(input_, begin, size, name=None)|取出部分数据|取得 axis 轴指定下标 indices 的值|
|tf.gather(params, indices, validate_indices=None, name=None, axis=0)|
|tf.split(value, num_or_size_splits, axis=0, num=None, name=‘split’)|切割原tensor|
|tf.tile(input, multiples, name=None)|指定每个维度重复多次|
|tf.pad(tensor, paddings, mode=‘CONSTANT’, name=None, constant_values=0)|填充值|
|tf.concat(values, axis, name=‘concat’)|在指定维度拼接|
|tf.stack(values, axis=0, name=”pack”)|在指定维度拼接|
|tf.unstack(value, num=None, axis=0, name=‘unstack’)|tf.stack 的反过程|
|tf.sequence_mask|mask|

### 转置
|函数|说明  |
|--|--|
|tf.reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None, name=None, seq_dim=None, batch_dim=None)|序列次序转置|
|tf.transpose(a, perm=None, name=‘transpose’, conjugate=False)|维度位置交换函数|















参考 [tensorflow常用函数以及技巧](https://seanlee97.github.io/2018/10/20/tensorflow%E5%B8%B8%E7%94%A8%E5%87%BD%E6%95%B0%E4%BB%A5%E5%8F%8A%E6%8A%80%E5%B7%A7/)






























参考：[TensorFlow是什么](http://c.biancheng.net/view/1880.html)