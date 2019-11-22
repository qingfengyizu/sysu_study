## update
2019.09.01： 初稿

## 引言
免费任务图（Task Graph for Free，TGFF）它可以根据用户的设置生成具有不同运算和通信关系的无环图，特别适用于任务映射和任务调度问题的理论分析。

## 下载和安装
**下载：** TGFF代码和说明文档的下载从网页[TGFF 3.6](http://ziyang.eecs.umich.edu/projects/tgff/index.html)，该版本是C++版本。
**安装：** 解压进入文件夹，执行make，编译生成可以执行文件TGFF。
**帮助：** 查看manual手册，或者直接运行tgff

## 运行tgff的一个简单例子
一般来说，TGFF读入参数配置文件tgffopt文件，根据配置产生网络。
运行命令 
>tgff [filename] 
> #例如 tgff simple

## 基本命令

    #产生基本任务图框架
    tg_cnt <int>:  #设置生成任务图的数量
    task_cnt <int> <int>:  #设置任务图中任务的平均个数和正负变动范围
    task_degree <int> <int>: #设置任务图中任务的最大输入和输出连接个数（degree）
    period_mul <list(<int>)>:#设置在多重速率的系统中，TGFF从列表中选择一个缩放周期时间。
    prob_periodic <flt>: # 设置周期和deadline的关系，默认值为1
    
    #关于并行的命令

	#产生相关的表，即与边相关的表
    table_label <string>:  #设置表格名字
    table_cnt <int>: #设置表格的数量
    task_attrib <list(<string> <flt> <flt> <flt>)>: #设置名字，均值，上下浮动范围
    最后加 trans_write 写到tgff file
	
	#输出
    tg_write: #将task graphs 写到tgff file
    pe_write: #将PE i信息写到tgff file
    trans_write: #将传输信息写到tgff file
    misc_write: #将依赖信息写到tgff file
    eps_write: #产生可视任务图
    vcg_write:  #产生可视VCG文件
    pack_schedule <int> <int> <flt> <flt> <flt> <flt> ... <int> <int> <int> <int> [<flt>] [to .tgff file & .eps file] [... on oneline]
    # num_task_graphs avg_task_graphs_per_pe avg_task_time mul_task_time task_slack task_round num_pe_types num_pe_soln num_com_types num_com_soln and optionally arc_fill_factor.
    
     
    
举例来说
 

```
table_label COMMUN
table_cnt 1
type_attrib data_size 100 20
trans_write

tg_cnt 1
task_cnt 8 1
task_degree 4 4
period_laxity 1
period_mul 1
tg_write
eps_write
vcg_write
```
    






## 参考
TGFF manual.pdf


