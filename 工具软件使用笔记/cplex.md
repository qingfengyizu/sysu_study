# CPLEX 笔记

参考文献
[IBM cplex 套件](https://www.ibm.com/support/knowledgecenter/zh/SSSA5P_12.7.0/ilog.odms.studio.help/Optimization_Studio/orientation_guide/topics/new_to_cos.html)
[cplex用户手册](https://www.ibm.com/support/knowledgecenter/zh/SSSA5P_12.7.0/ilog.odms.cplex.help/CPLEX/homepages/usrmancplex.html)


### 快速入门
CPLEX Studio 使用项目这一概念将一个 OPL 模型文件与一个或多个数据文件以及一个或多个设置文件相关联。 根文件夹中的项目文件组织所有相关模型、数据和设置文件。
IBM ILOG CPLEX Studio 使用项目这一概念将 OPL 模型 (.mod) 文件与（通常情况下）一个或多个数据 (.dat) 文件以及一个或多个设置 (.ops) 文件相关联。
仅包含单个模型文件的项目是有效的；数据和设置文件是可选的。 但是，一个项目可以包含多组模型、数据和设置文件，这些文件之间的关系由运行配置进行维护。
OPL 模型文件声明了数据元素，但是不一定将其初始化。 数据文件包含模型中声明的数据元素的初始化。


