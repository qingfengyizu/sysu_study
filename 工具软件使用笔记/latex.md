## 基本脚本

```
%基本脚本
%-*- coding: UTF-8 -*-
\documentclass[UTF8]{ctexart}
\title{RNN基础理论}
\date{2019.07.21}

%添加书签
\usepackage[breaklinks,colorlinks,linkcolor=black,citecolor=black,urlcolor=black]{hyperref}
%加入代码段
\usepackage{listings}
%添加图片
\usepackage{caption}
\usepackage{graphicx, subfig}
%添加图片路径
\graphicspath{{pic/}}
%图片编号依据章节
\usepackage{amsmath}
\numberwithin{figure}{section}
%数学符号
\usepackage{amssymb}
%公式中字体加粗
\usepackage{amssymb}
\usepackage{bm}
%超链接
\usepackage{hyperref}
\hypersetup{
	colorlinks=true,
	linkcolor=blue,
	filecolor=blue,      
	urlcolor=blue,
	citecolor=cyan,
}





%文章内容部分
\begin{document}
\maketitle
\tableofcontents

\end{document}


```

## 常用命令

```
%-------------------------------------------------------------------
%常用命令

空出一行
\vbox{}

%图片
%添加到正文中
\includegraphics[height=0.5cm]{formula6_4.png}
\centerline{\includegraphics[scale=0.3]{formula6_4.png}}

%超链接
\href{http://www.xiaoledeng.cn}{Xiao-Le Deng的博客}


%添加图片并引用
\ref{formula6_4}
\begin{figure}[htbp]
\centering
\includegraphics[scale=0.5]{formula6_4.png}
\caption{}
\label{formula6_4}
\end{figure}



%超链接
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=blue,      
    urlcolor=blue,
    citecolor=cyan,
}
\href{http://www.xiaoledeng.cn}{Xiao-Le Deng的博客}



%公式
公式字体加粗
\usepackage{bm}
\bm{x}
\boldsymbol{w}
公式字体斜体变正体
{\rm x_{z}}
数学公式中的省略号
\cdots
公式中的点乘
\cdot
求导
\partial 
```

## 在word中添加latex伪代码
1、添加package库。

```
\usepackage{amsmath}
\usepackage{amssymb}
% \usepackage{euler}
\providecommand{\abs}[1]{\left\lvert#1\right\rvert}
\providecommand{\norm}[1]{\left\lVert#1\right\rVert}
\usepackage{bbm}
\usepackage{CJK}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{amsmath,bm,graphicx,multirow,bm,bbm,amssymb,psfrag,algorithm,subfigure,color,mdframed,wasysym,subeqnarray,multicol}

\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{amsmath}
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}
```

2、添加伪代码块

```
\renewcommand{\thealgorithm}{1}
\begin{algorithm}[H] 
\caption{*******************************************} 
\label{ABCLFRS}
\begin{algorithmic}[1] 
\Require{S,$\lambda$,T,k} 
\Ensure{$\mathbf{w}_{222}$}\\ 
\textbf{initialize}: Set $\mathbf{w}_1 = 0$ 
\For{$t = 1,2,...,T$} 
\State Choose $A_t \subset[m]$
\EndFor
\end{algorithmic} 

\end{algorithm}
```
