# CPLEX 笔记

参考文献
[IBM cplex 套件](https://www.ibm.com/support/knowledgecenter/zh/SSSA5P_12.7.0/ilog.odms.studio.help/Optimization_Studio/orientation_guide/topics/new_to_cos.html)
[cplex用户手册](https://www.ibm.com/support/knowledgecenter/zh/SSSA5P_12.7.0/ilog.odms.cplex.help/CPLEX/homepages/usrmancplex.html)


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






































