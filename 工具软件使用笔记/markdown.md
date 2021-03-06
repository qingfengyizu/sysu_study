内容：markdown note

time : 2019.09.01 

[TOC]



## 常用

**插入表格**

```markdown
<div align=center>
<img width="500" src="pic\Figure 2.4 Tensors are views over a Storage instance.png" >
</div>	
```

**公式编号**

```markdown
Suppose we solve equations 
$$
\mathcal L U = F \tag{1}
$$
...
...
In the equation $\eqref{eq1}$, ...
```

**公式中的空格**

\quad



Markdown是一种纯文本格式的标记语言。通过简单的标记语法，它可以使普通文本内容具有一定的格式
文章主要是介绍Markdown的基本用法，方便后续写博文和脚本记录文档。

## 标题
一个#是一级标题，二个#是二级标题，以此类推。支持六级标题。
注：标准语法一般在#后跟个空格再写文字

## 字体
 要加粗的文字左右分别用两个 \* \*号包起来
要倾斜的文字左右分别用一个 \* 号包起来
要倾斜和加粗的文字左右分别用三个\*号包起来
要加删除线的文字左右分别用两个~~号包起来

## 引用
在引用的文字前加>即可。引用也可以嵌套，如加两个>>三个>>>

## 分割线
三个或者三个以上的 - 或者 * 都可以。

## 图片

## 超链接

    [超链接名](超链接地址 "超链接title")
    title可加可不加

```
<div align=center>
<img width="500" src="pic\Figure 2.4 Tensors are views over a Storage instance.png" >
</div>	
```



## 无序列表

无序列表用 - + * 任何一种都可以，注意：- + * 跟内容之间都要有一个空格。

    - 列表内容

##  有序列表

## 表格

    表头|表头|表头
    ---|:--:|---:
    内容|内容|内容
    内容|内容|内容
    
    第二行分割表头和内容。
    - 有一个就行，为了对齐，多加了几个
    文字默认居左
    -两边加：表示文字居中
    -右边加：表示文字居右
    注：原生的语法两边都要用 | 包起来。此处省略

表头|表头|表头
---|:--:|---:
内容|内容|内容
内容|内容|内容



### 表格换行和合并单元格

表格内容换行 `<br>` 

合并表格行 `colspan`

```html
<table>
	<tr>
		<td>小王</td>
		<td>小小王</td>
	<tr>
	<tr>
		<td colspan="2">隔壁老王</td>
	<tr>
</table>
```

合并表格列`rowspan`

```html
<table>
	<tr>
		<td>车神</td>
		<td>名车</td>
	</tr>
	<tr>
		<td rowspan="2">隔壁老王</td>
		<td>自行车</td>
	</tr>
	<tr>
		<td>电瓶车</td>
	</tr>
</table>
```

参考 [Markdown表格-换行、合并单元格](https://blog.csdn.net/qq_42711815/article/details/89257489)



## 代码

单行代码：代码之间分别用一个反引号包起来
`代码内容`
代码块：代码之间分别用三个反引号包起来，且两边的反引号单独占一行

    (```)
      代码...
      代码...
      代码...
    (```)


参考：
[Markdown基本语法](https://www.jianshu.com/p/191d1e21f7ed)
