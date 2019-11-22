111## 一、引言
### 1.1 shell脚本是什么
shell是一种紧凑而有效的编程语言，包含着测试、判断、循环、函数等基本功能，用shell脚本来管理重复性的测试任务。
##
### 1.2 一个简单的例子

```
#!/bin/bash
# Run as root, of course.
# Insert code here to print error message and exit if not root.

LOG_DIR=/var/log
# Variables are better than hard-coded values.
cd $LOG_DIR

cat /dev/null > messages
cat /dev/null > wtmp


echo "Logs cleaned up."

exit #  The right and proper method of "exiting" from a script.
     #  A bare "exit" (no parameter) returns the exit status
     #+ of the preceding command. 
```


  shell脚本的第一行为sha-bang (#!)，作用是选择合适的编译器，如在Linux中一般用`/bin/bash`。

    #!/bin/sh
    #!/bin/bash
    #!/usr/bin/perl
    #!/usr/bin/tcl
    #!/bin/sed -f
    #!/bin/awk -f


## 基础语法

### 2.1 特殊字符
**#**	
一般作为注解符号。
**；** 
用来实现多条命令写在同一行中，例如`if [ -x "$filename" ]; then    #  Note the space after the semicolon.`
**;;**  
在case中作为终止符号。
**.**
用点来表示执行，等价于source操作，也常用来表示当前路径，等价于pwd操作。
**!**
状态取反。
**? **
用于三目运算中，测试操作。`condition?result-if-true:result-if-false`
**$**
表示变量替换。`echo $var1`  
**()**
数组初始，`Array=(element1 element2 element3)`
**[ ]**
表示测试，
**$[ ... ]   (( ))**  
整数计算

    a=3
    b=7
    echo $[$a+$b]   # 10
    echo $[$a*$b]   # 21

**> &> >& >>**
将输出写入到文本中

## 2.2 参数和变量
变量的赋值，初始化为数值或者字符串。在bash中所有的变量都是**untyped**的，字符串转为取整为0。
变量的引用，需要区分单引号(' ')、特殊符号$ 和双引号(" ")，单引号是指简单字符串，双引号会将字符串中的变量和转移字符做替换。常用的转义字符包括`\" \$  \\ `

    variable1=23
    category=minerals  				# No spaces allowed after the "=".
    number=$RANDOM					#random	
    echo $variable1
    echo ${variable1}
    echo "\$var1 = "$var1""      	# $var1 = Two bits
    echo -e "\v\v\v\v"   		    # Prints 4 vertical tabs.
    echo $'\n'           			# Newline.
    echo $?    						# Exit status 0 returned because command executed successfully.
    unset variable1
带参数的脚本输入输入如下，脚本中引用这些变量，按照输入的顺序分别为 $0, $1, $2, $3。


    # Call this script with at least 10 parameters, for example
    # ./scriptname 1 2 3 4 5 6 7 8 9 10


### 2.3 判断（if/then）
if/else 判决结构的基本框架如下，[]是基本判断方式，[[ ]]推荐用来test，(( ))推荐用来算术test。
```
if [ condition1 ]
then
   command1
   command2
   command3
elif [ condition2 ]
# Same as else if
then
   command4
   command5
else
   default-command
fi
```
### 2.4 循环结构
 for，while，until循环基本结构如下，

```
for arg in [list]
do 
 command(s)... 
done
```

```
while [ condition ]
do 
 command(s)... 
done
```

```
until [ condition-is-true ]
do 
 command(s)... 
done
```
### 2.5 分支case

```
case "$variable" in 

 "$condition1" ) 
 command... 
 ;; 

 "$condition2" ) 
 command... 
 ;; 
```

### 2.5 命令替换
用来替换命令的输出结果，操作格式`(`...`)`或`$(...)`

```

script_name=`basename $0`
echo "The name of this script is $script_name."
echo $(echo \\)
```

参考：
[Advanced Bash-Scripting Guide An in-depth exploration of the art of shell scripting](http://tldp.org/LDP/abs/html/) 
