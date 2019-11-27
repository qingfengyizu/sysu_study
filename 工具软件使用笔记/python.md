# 一、Python基础
## 1.1 基本语法
**行与缩进：**  python最具特色的就是使用缩进来表示代码块，不需要使用大括号 {} 。
**多行语句：** Python 通常是一行写完一条语句，但如果语句很长，我们可以使用反斜杠(\)来实现多行语句，在 [], {}, 或 () 中的多行语句，不需要使用反斜杠(\\)。
**数字类型：** 整数int、布尔型bool(true)、浮点数float和复数complex(1+2j、1.1+2.2j)。
**字符串：** 引号和双引号使用完全相同，使用三引号('''或""")可以指定一个多行字符串。字符串可以用 + 运算符连接在一起，用 * 运算符重复。字符串有两种索引方式，从左往右以 0 开始，从右往左以 -1 开始。
**同一行显示多条语句：**  语句之间使用分号(;)分割
**Print 输出：** 需要在变量末尾加上 end=""：
**import 与 from...import：** 导入相应的模块。

##  1.2 基本数据类型
Python3 中有六个标准的数据类型：
- Number（数字）
- String（字符串）
- List（列表）
- Tuple（元组）
- Set（集合）
- Dictionary（字典）
### 数字
Python3 支持 int、float、bool、complex（复数）。
通过使用del语句删除单个或多个对象。例如：del var
基本数字运算包括：加+，减-，乘\*，除法得到浮点数/，除法得到整数//，取余数%，乘方 \*\* 。
### 字符串
Python中的字符串用单引号 ' 或双引号 " 括起来，同时使用反斜杠 \ 转义特殊字符。
Python中的字符串不能改变。
字符串可以用+运算符连接在一起，用*运算符重复。
Python 使用反斜杠(\)转义特殊字符，如果你不想让反斜杠发生转义，可以在字符串前面添加一个 r，表示原始字符串，例如print(r'Ru\noob')
 函数|描述  |
|--|--|
|      	len(string)|           返回字符串长度         |

### List（列表）
列表是写在方括号 [] 之间、用逗号分隔开的元素列表。列表可以完成大多数集合类的数据结构实现，是 Python 中使用最频繁的数据类型。
加号 + 是列表连接运算符，星号 * 是重复操作，这与字符串相似，与Python字符串不一样的是，列表中的元素是可以改变的。
 方法|描述  |
|--|--|
|     list.append(obj)      |         在列表末尾添加新的对象         |
|     list.count(obj)       |           统计某个元素在列表中出现的次数           |
|     list.index(obj)       |      从列表中找出某个值第一个匹配项的索引位置                |
|      	list.insert(index, obj)      |       将对象插入列表               |
|       list.pop([index=-1])     |             移除列表中的一个元素（默认最后一个元素），并且返回该元素的值         |
|     list.remove(obj)       |        移除列表中某个值的第一个匹配项              |
|        list.reverse()    |         反向列表中元素             |
|  list.sort( key=None, reverse=False)          |             对原列表进行排序         |
|       list.clear()     |        清空列表             |

### Tuple（元组）
元组写在小括号 () 里，元素之间用逗号隔开。元组（tuple）与列表类似，不同之处在于元组的元素不能修改。其实，可以把字符串看作一种特殊的元组。
 方法|描述  |
|--|--|
|      	len(tuple)      |             计算元组元素个数。         |
|    	max(tuple)        |                      |
|       min(tuple)     |                      |
|  	tuple(seq)          |         将列表转换为元组。             |

### Set（集合）
使用大括号 { } 或者 set() 函数创建集合，注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典。集合（set）是由一个或数个形态各异的大小整体组成的，构成集合的事物或对象称作元素或是成员。
 方法|描述  |
|--|--|
|        s.add( x )    |              添加元素        |
|    s.update( x )        |      添加元素                |
|     s.remove( x )       |         移除元素             |
|    s.discard( x )        |         移除元素             |
|    len(s)        |             计算集合元素个数         |
|   s.clear()         |            清空集合          |
|      x in s      |            判断元素是否在集合中存在          |

### Dictionary（字典）
字典是一种映射类型，字典用 { } 标识，它是一个无序的 键(key) : 值(value) 的集合。列表是有序的对象集合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。
构造函数 dict() 可以直接从键值对序列中构建字典dict([('Runoob', 1), ('Google', 2), ('Taobao', 3)])，或者{'Taobao': 3, 'Runoob': 1, 'Google': 2}
### 数据类型转换
float(x)  将x转换到一个浮点数
str(x) 将对象 x 转换为字符串
hex(x) 将一个整数转换为一个十六进制字符串
## 1.3 运算
### 成员运算符
测试实例中包含了一系列的成员，包括字符串，列表或元组。
in	如果在指定的序列中找到值返回 True，否则返回 False。
not in	如果在指定的序列中没有找到值返回 True，否则返回 False。
### 常用数学函数
| 函数|描述  |
|--|--|
| abs(x) | 绝对值 |
| ceil(x) | 上取整 |
| exp(x) | 求指数 |
| foor(x) | 下取整 |
| log(x) |  |
| log10(x) | 绝对值 |
| round(x) | 四舍五入 |
| sqrt(x) | 平方根 |
### 随机数函数
| 函数|描述  |
|--|--|
|      choice(seq)     |            从序列的元素中随机挑选一个元素         |
|     randrange ([start,] stop [,step])      |         从指定范围内，按指定基数递增的集合中获取一个随机数，基数默认值为 1            |
|      random()     |          随机生成下一个实数，它在[0,1)范围内。           |
|     shuffle(lst)      |            将序列的所有元素随机排序         |
|     uniform(x, y)      |        随机生成下一个实数，它在[x,y]范围内。             |
按概率随机选取：np.random.choice([0, 1, 2, 3], p=[0.1, 0.0, 0.7, 0.2])


## 1.4 控制块
### 条件控制
Python 中用 elif 代替了 else if，所以if语句的关键字为：if – elif – else。
### 循环语句
while 循环 
for 语句   for \<variable> in <sequence>:    一般用range()函数生成序列。
break 语句可以跳出 for 和 while 的循环体。
pass是空语句，是为了保持程序结构的完整性。
## 1.5 函数
**定义函数** 函数代码块以 def 关键词开头，后接函数标识符名称和圆括号 ()。
**传递变量** 不可变，如 整数、字符串、元组。可变类型：列表，字典。如 fun（la），则是将 la 真正的传过去，修改后fun外部的la也会受影响

## 1.6 输入和输出
| 函数|描述  |
|--|--|
|     str()    |           输出函数     |
|   repr()   |       输出函数          |
|  open(filename, mode)      |       打开文件          |
|  f.read()    |         读取一个文件的内容        |
|   f.readline()   |        文件中读取单独的一行         |
|    f.readlines()  |        返回该文件中包含的所有行。         |
|   f.write()   |  将 string 写入到文件中               |
|   f.seek(offset, from_what)    |                 改变文件当前的位置|
| f.close()     |           关闭文档      |
	
	# 举例写写文本
	save_text_fid = open(save_text_path, 'w')
	save_text_fid.write("%10.2f   %15.6f \n" % (reward ,softmax  ))
	save_text_fid.close()
	# 读文本
	line_raw = sequence_lable_fid.readline()
	if not line_raw:
		break
 	line_data = np.array(list(map(int, line_raw.split())))
	sequence_lable_fid.close()


## 1.7 添加模块路径
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
参考： [Python3](https://www.runoob.com/python3/python3-basic-syntax.html)

## 1.8 排列组合
用到combinations和permutations两个函数。
```python
import itertools
print list(itertools.combinations([1,2,3,4],4)) #不考虑排列
print list(itertools.permutations([1,2,3,4],4)) #考虑排列，全排列
```




# 二、Numpy
NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。NumPy 通常与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用， 这种组合广泛用于替代 MatLab，是一个强大的科学计算环境，有助于我们通过 Python 学习数据科学或者机器学习。
### 创建Ndarray
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
例如： 
np.array([1,2,3])       
np.array([1,  2,  3,4,5], ndmin =  2)       
np.array([1,  2,  3], dtype = complex)


| 创建矩阵方法|描述  |
|--|--|
|    numpy.empty(shape, dtype = float, order = 'C')      |         创建未初始化的数组         |
|     numpy.zeros(shape, dtype = float, order = 'C')      |          创建指定大小的数组，数组元素以 0 来填充         |
|      numpy.ones(shape, dtype = None, order = 'C')     |          创建指定形状的数组，数组元素以 1 来填充         |
|    numpy.arange(start, stop, step, dtype)       |           根据 start 与 stop 指定的范围以及 step 设定的步长，生成一个 ndarray        |
|    np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)       |      创建一个一维数组，数组是一个等差数列构成的             |
|      np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)     |             用于创建一个于等比数列      |



| 属性|描述  |
|--|--|
|    ndarray.ndim       |          秩，即轴的数量或维度的数量         |
|       ndarray.shape    |         数组的维度，对于矩阵，n 行 m 列          |
|     ndarray.size      |           数组元素的总个数，相当于 .shape 中 n*m 的值        |
| numpy.reshape(arr, newshape, order='C')| 不改变数据的条件下修改形状|
|numpy.ndarray.flat|数组元素迭代器|
|ndarray.flatten(order='C')|返回一份数组拷贝，对拷贝所做的修改不会影响原始数组|
|numpy.ravel(a, order='C')|展平的数组元素，顺序通常是"C风格"，修改会影响原始数组|
|numpy.transpose(arr, axes)|对换数组的维度|
|numpy.expand_dims(arr, axis)|在指定位置插入新的轴来扩展数组形状|
|numpy.squeeze|从给定数组的形状中删除一维的条目|
|numpy.concatenate((a1, a2, ...), axis)|函数用于沿指定轴连接相同形状的两个或多个数组|
|numpy.stack(arrays, axis)|沿新轴连接数组序列|
|numpy.hstack| numpy.stack 函数的变体，它通过水平堆叠来生成数组|
|numpy.vstack|numpy.stack 函数的变体，它通过垂直堆叠来生成数组|
|numpy.split(ary, indices_or_sections, axis)|沿特定的轴将数组分割为子数组|
|numpy.hsplit|水平分割数组|
|numpy.vsplit|沿着垂直轴分割|
|numpy.append(arr, values, axis=None)|函数在数组的末尾添加值|
|numpy.insert(arr, obj, values, axis)|函数在给定索引之前，沿给定轴在输入数组中插入值。|、
|Numpy.delete(arr, obj, axis)|从输入数组中删除指定子数组的新数组|

### Numpy 字符串函数
|函数|描述  |
|--|--|
| add() | 对两个数组的逐个字符串元素进行连接 |
|multiply()|返回按元素多重连接后的字符串|
|center()|居中字符串|
|title()|将字符串的每个单词的第一个字母转换为大写|
|lower()|数组元素转换为小写|
|upper()|	数组元素转换为大写|
|split()|指定分隔符对字符串进行分割，并返回数组列表|

### Numpy数学函数
### 随机
np.random.shuffle(b)

#### 三角函数
NumPy 提供了标准的三角函数：sin()、cos()、tan()。
arcsin，arccos，和 arctan 函数返回给定角度的 sin，cos 和 tan 的反三角函数。
numpy.around() 函数返回指定数字的四舍五入值。
#### 计算函数
简单的加减乘除: add()，subtract()，multiply() 和 divide()。
numpy.reciprocal() 函数返回参数逐元素的倒数。
numpy.power() 函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂。
numpy.mod() 计算输入数组中相应元素的相除后的余数。 函数 numpy.remainder() 也产生相同的结果。
#### 统计函数
numpy.amin() 用于计算数组中的元素沿指定轴的最小值。
numpy.amax() 用于计算数组中的元素沿指定轴的最大值。
numpy.ptp()函数计算数组中元素最大值与最小值的差（最大值 - 最小值）。
numpy.median() 函数用于计算数组 a 中元素的中位数（中值）
numpy.mean() 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。
numpy.average() 函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值。
numpy.std 标准差
numpy.var 方差

#### 线性代数
numpy.dot() 对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为内积)
numpy.vdot() 函数是两个向量的点积。
numpy.inner() 函数返回一维数组的向量内积。
numpy.matmul 函数返回两个数组的矩阵乘积。
numpy.linalg.det() 函数计算输入矩阵的行列式。
numpy.linalg.solve() 函数给出了矩阵形式的线性方程的解。
numpy.linalg.inv() 函数计算矩阵的乘法逆矩阵。

#### IO
savetxt() 函数是以简单的文本文件格式存储数据，对应的使用 loadtxt() 函数来获取数据。
np.loadtxt(FILENAME, dtype=int, delimiter=' ')
np.savetxt(FILENAME, a, fmt="%d", delimiter=",")



#### 广播
广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行。当运算中的 2 个数组的形状不同时，numpy 将自动触发广播机制。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190903152949466.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXN0bnV0c3M=,size_16,color_FFFFFF,t_70)
参考 [Numpy](https://www.runoob.com/numpy/numpy-ndarray-object.html)

### 常用功能
#### 实现onehot

    arr = np.arange(seq_length)
    np.random.shuffle(arr)
    x_one_hot = np.eye(seq_length)[arr]
#### 保存数据

```python
np.loadtxt()
np.savetax()

import scipy.io as sio
data=sio.loadmat('saveddata.mat')
sio.savemat('saveddata.mat', {'xi': xi,'yi': yi,'ui': ui,'vi': vi})
```

# 三、Matplotlib
一个简单的例子

```python
import matplotlib
matplotlib.use('Agg')#关闭gui
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

plt.plot(x, y, '.', c='r')
plt.xlabel("p ")  # x轴上的名字
plt.ylabel("c ")  # y轴上的名字
plt.xticks(rotation=10) # 旋转
plt.yticks(rotation=10)
# plt.show()
fig_path = 'name.jpg'
plt.savefig(fig_path)
```

# 四、Jupyter
## 修改Jupyter Notebook默认启动目录
### 4.1 直接通过 Jupyter Notebook 的快捷方式进入
首先找到Jupyter Notebook的快捷方式。
“目标”中最后一个参数默认是%USERPROFILE%/，就是默认启动目录。
将其修改为对应的路径，注意路径为为**反斜杠**。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190904094601286.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZXN0bnV0c3M=,size_16,color_FFFFFF,t_70)

### 4.2 通过anaconda进入
在anaconda prompt命令窗口中输入 

    jupyter notebook --generate-config

这个命令的作用是生成 Jupyter notebook 的配置文件，主要目的只是为了找到这个文件的路径。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190915192638471.png)
找到 jupyter_notebook_config.py 的路径并打此文件。

找到 c.NotebookApp.notebook_dir 这个变量，将你希望的路径赋值给这个变量，并删除这一行前面的“#”。修改后如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190915192718273.png)
改完后保存。再次通过 Anaconda Navigator 进入 jupyter notebook 的时候会发现默认路径已经更改。