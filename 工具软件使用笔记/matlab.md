#### 得到当前时间的字符串

    'Time_',datestr(now,30)
#### 画图基本操作

```matlab
figure
x =pmw;
y = communication_cost;
plot(x,y, '.r');
xlabel('$p\left({M} | {W}\right) $','interpreter','latex' ) %latex代表表示公式
ylabel("communication cost mb/s")
print(gcf,[fold_path, 'pwd_communication_node9_before_train.jpg'],'-djpeg', '-r300') %设置分辨率
saveas(gcf,[fold_path, 'pwd_communication_node9_before_train.fig']);
```



      save([dir_path, 'data.mat'],'data') ;

#### 随机相关函数

```matlab
% 依据概率随机选择
a=[1 2 3 4 5]
Prob=[0.31 0.07 0.33 0.09 0.2]
S=randsrc(1,1,[a;Prob])
```



#### 统计某一行出现的次数

```matlab
a=[1,2; 3,4; 2,3; 1,2; 3,4;1,2;1,2;3,4;2,3]
[b, m, n] = unique(a,'rows');
tbl = tabulate(n);
table = zeros(length(m),3 );
table(:, 1:2) = b;
table(:,3) = tbl(:, 2 );
```

