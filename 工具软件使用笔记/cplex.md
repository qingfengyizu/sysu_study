# CPLEX 笔记

参考文献
[IBM cplex 套件](https://www.ibm.com/support/knowledgecenter/zh/SSSA5P_12.7.0/ilog.odms.studio.help/Optimization_Studio/orientation_guide/topics/new_to_cos.html)
[cplex用户手册](https://www.ibm.com/support/knowledgecenter/zh/SSSA5P_12.7.0/ilog.odms.cplex.help/CPLEX/homepages/usrmancplex.html)

[IBM ILOG CPLEX Optimization Studio Getting Started with CPLEX](CPLEX124_gscplex.pdf)  

### 快速入门

CPLEX Studio 使用项目这一概念将一个 OPL 模型文件与一个或多个数据文件以及一个或多个设置文件相关联。 根文件夹中的项目文件组织所有相关模型、数据和设置文件。

IBM ILOG CPLEX Studio 使用项目这一概念将 OPL 模型 (.mod) 文件与（通常情况下）一个或多个数据 (.dat) 文件以及一个或多个设置 (.ops) 文件相关联。

仅包含单个模型文件的项目是有效的；数据和设置文件是可选的。 但是，一个项目可以包含多组模型、数据和设置文件，这些文件之间的关系由运行配置进行维护。

OPL 模型文件声明了数据元素，但是不一定将其初始化。 数据文件包含模型中声明的数据元素的初始化。

OPL 项目的根文件夹中的 .project 文件组织所有相关模型、数据和设置文件。 运行配置（在 .oplproject 文件中维护）也提供了一种便捷方法来维护环境的相关文件与运行时选项之间的关系，

一个典型项目包含：

 - 一个或多个 OPL 模型文件
 - 任何数量的数据文件，或者无数据文件
 - 任何数量的设置文件，或者无设置文件
 - 引用这些模型、数据和设置文件的各种组合的一个或多个运行配置。 （一个运行配置不能具有一个以上的模型文件。

 #### OPL模型
 **数据的声明**数据声明允许您对数据项命名，这样就可在模型中轻松引用这些数据项。

 **决策变量的声明**在 OPL 上下文中，变量是决策变量，这与 IBM ILOG Script 和常规编程上下文不同。 决策变量的声明将为模型中的每个变量命名并提供其类型。 

**目标函数**目标函数是您希望优化的函数。 此函数必须由您先前在模型文件中声明的变量和数据组成。 目标函数由 minimize 或 maximize 关键字引入。 

**约束**约束指示模型的可行解法所需的条件。 您在 subject to 块中声明约束。

#### 数据文件
通过将问题的模型与实例数据分离，您可以更好地组织大型问题。 每组数据都存储在一个单独数据文件中。可通过将问题的模型从实例数据分离来更好地组织大型问题。 每组数据都存储在一个单独数据文件中，并且扩展名为 .dat。

可扩展数据，请注意容量计算中整数除法运算符 div 的使用以及注意求模运算符 mod 的使用。

#### 配置文件
如果您更改设置的缺省值，那么用户定义的新值将存储在与项目相关联的设置文件中。设置文件 (.ops) 是当您决定更改 OPL 语言选项、约束规划 (CP Optimizer) 参数或数学规划 (CPLEX) 参数的缺省值时，用于存储用户定义的值的位置。


#### 运行
需要特别注意的是，win10下默认配置名是中文，需要改为英文命名，否则会报错。

参考lab ：
Program Files (x86)\CPLEX_Studio128\cplex\examples\src


### 2.1 python 调用CPLEX

#### 2.1.1 设置CPLEX的Python API接口



##### Linux
要在系统上安装 CPLEX-Python 模块，请使用位于 yourCplexhome/python/VERSION/PLATFORM 中的脚本 setup.py。 如果要将 CPLEX-Python 模块安装在非缺省位置，请使用选项 --home 识别安装目录。 例如，要将 CPLEX-Python 模块安装在缺省位置，请从命令行使用以下命令：

``` 
python3 setup.py install --home /usr/local/cplex-python
export PYTHONPATH=/usr/local/CPLEX_Studio129/cplex/python/3.6/x86-64_linux/
```
参考 [设置CPLEX的Python API](https://www.ibm.com/support/knowledgecenter/zh/SSSA5P_12.7.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html)

##### win
在win下，将C:\Program Files\IBM\ILOG\CPLEX_Studio128\cplex在里面的python文件夹下进入对应python版本的子文件夹，再复制里面的cplex文件夹到本地python的site package文件夹内，即可使用例如在我的电脑上，是复制C:\Programiles\IBM\ILOG\CPLEX_Studio128\cplex\python\3.6\x64_win64\cplex到D:\ProgramData\Anaconda3\envs\py27\Lib\site-packages当中。

参考[【python】windows下python调用cplex](https://blog.csdn.net/weixin_42437606/article/details/80657240)







# Python tutorial

##### 安装（win）

进入/IBM/ILOG/CPLEX_Studio128/cplex/目录的python子目录下，执行python setup.py install。[参考](http://lyminghao.site/cplex-python-api/)

##### 访问cplex

```python
import cplex
```

参考脚本在 **CPLEX_Studio128\cplex\examples\src\python** 下

##### 构建和解决小的LP

有三种方式

- model by rows  
- model by columns  
- model by nonzero elements  

参考脚本 lpex1.py  用着三种不同的方法来解决LP。

##### 读和写cplex模型

将模型写到LP格式的文件中。

```python
 my_prob.write("lpex1.lp")
```

读LP格式文件

```python
my_prob.write("lpex1.lp")
```

##### 选择优化器

cplex默认根据模型的特点选择恰当的求解方法。当然也可以修改某些模型的参数。

```python
c = cplex.Cplex(filename)
alg = c.parameters.lpmethod.values
if method == "o":
    c.parameters.lpmethod.set(alg.auto)
elif method == "p":
    c.parameters.lpmethod.set(alg.primal) #primal simplex optimizer   
elif method == "d":
    c.parameters.lpmethod.set(alg.dual) #dual simplex optimizer    
```



##### 参数的设置和查询

parameter提供一下几种方法：

- get() 返回当前参数的值.
- set(value) 设置参数值为value，当value值超出范围则报错。
- reset() 将参数设置为默认值
- default() 返回默认参数值
- type() 返回参数类型
- help() 给出参数的简单描述 

数值参数提供两种额外的方法：

- min() 返回参数允许的最小值
- max() 返回参数允许的最大值  

某些整数，例如lpmethod ，具有特殊的意义。

例如，添加一个新的变量，并设置其上界

```python
 c.variables.add(names = ["new_var"], ub = [1.0])
```

获得该变量的索引

```python
c.variables._get_index("newvar")
```

获得该变量的上界，当然也可以用索引值来代替变量名

```python
c.variables.get_lower_bounds('newvar')
c.variables.get_lower_bounds(0)
c.variables.get_upper_bounds('newvar')
```

重新设置变量的上界

```python
c.variables.set_upper_bounds('newvar',2)
```

可以利用索引来获得多个变量的上边界

```python
c.variables.add(names=["var1","var2"], ub =[1.0,2.0])
c.variables.get_upper_bounds(0,2)
c.variables.get_upper_bounds()
```

##### 结果信息的查询



```
>>> c = cplex.Cplex()
>>> c.parameters.tuning.timelimit.set(300.0)
>>> c.parameters.tuning.timelimit.get()
300.0
```



# Docplex

##### 安装

[参考](https://zhuanlan.zhihu.com/p/54894350)

```python
pip install docplex
```

##### docplex 类

[参考](http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html)



python


