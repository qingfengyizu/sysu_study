##什么是 Github?
github是一个基于git的代码托管平台，付费用户可以建私人仓库，我们一般的免费用户只能使用公共仓库，也就是代码要公开。

##Github 安装
下载 git Windows版

##配置Git
首先在本地创建ssh key；

    $ ssh-keygen -t rsa -C "your_email@youremail.com"

后面的your_email@youremail.com改为你在github上注册的邮箱，之后会要求确认路径和输入密码，我们这使用默认的一路回车就行。成功的话会在~/下生成.ssh文件夹，进去，打开id_rsa.pub，复制里面的key。

回到github上，进入 Account Settings（账户配置），左边选择SSH Keys，Add SSH Key,title随便填，粘贴在你电脑上生成的key。

为了验证是否成功，在git bash下输入：

    $ ssh -T git@github.com
如果是第一次的会提示是否continue，输入`yes`就会看到：You've successfully authenticated, but GitHub does not provide shell access 。这就表示已成功连上github。

接下来我们要做的就是把本地仓库传到github上去，在此之前还需要设置username和email，因为github每次commit都会记录他们。

    $ git config --global user.name "your name"
    $ git config --global user.email "your_email@youremail.com"

参考：[Github 简明教程](https://www.runoob.com/w3cnote/git-guide.html)

## 本地文件夹同步到GitHub
1、在github上创建项目

2、使用git 克隆到本地。

    git clone https://github.com/qingfengyizu/sysu_study.git



3、编辑项目


4、本地同步到GitHub

	git add . 
	git commit -m "提交说明"
	git pull origin master
	git push origin master