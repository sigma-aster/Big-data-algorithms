# clustering and classification

## 聚类的方法

### 基于中心的聚类

k-means/k-median/k-centers

### 谱聚类

映射到一个新的空间然后在k-means

### 密度聚类

聚类成为一个高密度的区域

### 层次聚类

不用给定需要聚类的数量k

分层聚类，根据上一层聚到下一层(分为自下而上合并和自上而下)

## 谱聚类

### Similarity Graphs

使用图来表现相似度矩阵

带权的邻接矩阵$W$，度矩阵$D$

给定一个数据点集合，相似度$s_{ij}$，距离$d_{ij}$，如何构建一个图：

1.$\epsilon$图。连接相似度小于$epsilon$的

2.k-nearest图。连接$v_i,v_j$如果$v_j$是$v_i$的k-nearest。仅当$v_i$和$v_j$都是对方的$k-nearest$

3.全连接图。

### Graph Laplacians

$L = D - W$​

#### 命题1

- 对于任意的$f \in R^n$，都有$f^{'}Lf = \frac{1}{2}\sum_{i,j}^nw_{i,j}(f_i-f_j)^2$

- 对称且半正定。

- 最小的特征值是0，对应的特征向量是$[1,1,\dots,1,1]$​

- 有n个非负特征值，记作$0 \leq \lambda_1 \leq \lambda_2 \cdots \leq \lambda_n$

#### 命题2

如果一个无向图有 $k$ 个连通分量，那么拉普拉斯矩阵将有 $k$ 个特征值为0。

- 每个连通分量对应于一个零特征值。

- 每个零特征值对应于一个特征向量，这些特征向量在对应的连通分量上有相同的值。

  证明: 

  ​	k=1时。要使得$0=f^{'}Lf = \frac{1}{2}\sum_{i,j}^nw_{i,j}(f_i-f_j)^2$，必须所有$f_i$都相等

  ​	k不为1时，将$L$分块。每一块是一个连通子图。

### 谱聚类算法

输入数据点和相似度函数

- 构建相似图$G$

- 计算标准拉普拉斯矩阵$L$
- 计算前$k$个特征向量$u_1,\cdots,u_k$
- $U \in R^{n\times k}$作为以$u_1,\cdots,u_k$作为列的矩阵
- 记$y_i$为$U$的第$i$行
- 使用k-means将$y_i$聚为k类$c_1,\cdots,c_k$

输出结果

**将聚类的目标从$x_i \in R^n$转变为了$y_i \in R^k$**

### Graph cut point of view

谱聚类$\rightarrow$一个图像分割问题的近似解

**mincut**

$mincut(A_1,\cdots ,A_k) = \frac{1}{2}\sum_i^kw(A_i,\overline{A_i})$

mincut经常把图分成单独的顶点，而聚类要求结果相对较大。

$RatioCut(A_1,\cdots ,A_k) = \frac{1}{2}\sum_i^k \frac{w(A_i,\overline{A_i})}{|A_i|}$

$Ncut(A_1,\cdots ,A_k) = \frac{1}{2}\sum_i^k \frac{w(A_i,\overline{A_i})}{vol(A_i)}$

松弛Ncut得到标准的谱聚类，松弛RatioCut得到不标准的谱聚类。

#### 当k=2时，近似RatioCut

目标 $minRatioCut(A,\overline A)$

定义$f = (f_1,\cdots,f_n)^{'}$，其中

$f_i = \sqrt{\frac{|\overline A|}{|A|}}$   if   $v_i \in A$

$f_i = -\sqrt{\frac{|A|}{|\overline A|}}$   if   $v_i \in \overline A$

利用$L$重写目标函数

$\begin{split} f^{'}Lf &= \frac{1}{2}\sum_{i,j}^n w_{i,j}(f_i-f_j)^2 \\ &= \frac{1}{2}\sum_{i \in A,j \in \overline A}^n w_{i,j}(\sqrt{\frac{|\overline A|}{|A|}} + \sqrt{\frac{|A|}{|\overline A|}})^2 + \frac{1}{2}\sum_{j \in A,i \in \overline A}^n w_{i,j}(\sqrt{\frac{|\overline A|}{|A|}} + \sqrt{\frac{|A|}{|\overline A|}})^2 \\ &= cut(A,\overline A)\cdot (\frac{|\overline A|}{|A|}+\frac{|A|}{|\overline A|}+2) \\ &= cut(A,\overline A)\cdot (\frac{|\overline A| + A}{|A|}+\frac{|A| + |\overline A|}{|\overline A|}) \\ &= |V|\cot Ratiocut(A,\overline A) \end{split}$

 此外还有

$\sum_i^n f_i =0, ||f||^2 = n$

这样Rationcut被重写成最小化$f^{'}Lf$。这仍然是一个NP-hard的问题

Rayleihj-Ritz定理说明$f$是矩阵$L$第二小的特征值对应的特征向量(最小的特征值是0)。

在有了$f$之后，可以聚类成两组$C,\overline C$

#### 对于任意的k

给定顶点$V$的一组划分，分成k类$A_1,A_2,\cdots,A_k$。定义指示向量$h_j = (h_{1,j},\cdots,h_{n,j})$

$h_{i,j} = \frac{1}{\sqrt{|A_j|}}$  if  $v_i \in A_j$   $(i =1,\cdots,n;j =1,\cdots,k)$

有$H^{'}H=I$,  $h_i^{'}Lh_i = \frac{cut(A_i,\overline{A_i})}{|A_i|}$

$RatioCut(A_1,\cdots,A_k) = \sum_i^k h_i^{'}Lh_i =Tr(H_i^{'}LH_i)$

这样Rationcut被重写成最小化$Tr(H_i^{'}LH_i)$，约束条件是$H^{'}H=I$。

和$k=2$的情况类似，我们将这个问题松弛成H可以取任意实数。

这是一个最小化迹问题的标准形式。

Rayleihj-Ritz定理说明H是包含$L$的前k个特征向量的矩阵，然后对U中的行使用k-means。

## 基于密度的聚类

*核心点：将在同一个密度区域的数据点聚类*

### Preliminary

- $\epsilon - neighborhood$

  给定数据点p，他的$\epsilon - neighborhood$定义为$N_{\epsilon}(P)=\{d(p,q)\leq \epsilon \}$​

- core point

  给定正整数M，如果$|N_{\epsilon}(P)|\geq M$。我们称p为$(\epsilon,M)$下的核心点

- border point

  给定核心点p，如果q不是核心点但是p的$\epsilon-neighborhood$，称q为边缘点

- Directly reachable

  如果$d(p,q) \leq \epsilon$，称p,q是直接可达的

- Reachable

  如果存在一条通路$p_1,\cdots,p_n$，其中$p_{i+1}$对$p_i$是直接可达的。称$p_1$和$p_n$是可达的

- outliers

  对所有其他点都不可达的称为noise points

划分在同一类中的点称为density-conncted的

### DBSCAN

**Idea**

- 对于每个点找出它的$\epsilon - neighborhood$，标出核心点
- 找到所有核心点的$\epsilon - neighborhood$，忽略非核心点
- 将在$\epsilon - neighborhood$中的非核心点聚类，否则划为噪声点

**query-based  algo**

输入$D, \epsilon, M$

1. 随机选取一个没有访问过的点p

2. 寻找p的$\epsilon - neighborhood$，如果$|N_{\epsilon}(P)|\leq M$，标记为outliers。否则将p作为第一类的核心，标记为已访问

3. 找到p所有的$\epsilon - neighborhood$，将其放入第一类并标记为已访问

4. 对于第一类中的所有点，重复2，3

随机挑选点执行2，3，4直到所有点被访问

优点：

- 不需要给定类别数k
- 对孤立点不敏感
- 可以聚类非线性数据

缺点：

- 需要给定$\epsilon$
- 不能聚类高维数据

### DBSCAN revisited

- DBSCAN 平均时间$O(nlogn)$，最坏情况$O(n)$
- 在二维空间存在$O(nlog(n))$算法
- $d \geq 3$时，需要$O(n^{\frac{4}{3}})$的算法

#### Algorithm in 2D space

将空间划分成$\frac{\epsilon}{\sqrt{2}} \times \frac{\epsilon}{\sqrt{2}}$的网格。包含了核心点的网格成为核心格。

#### DBSCAN in $\geq$ 3 dimensions

看不懂，等会儿回来再看

## 层次聚类

### Basic definitions

聚类序列$c_n,\cdots,c_1$，其中$|c_k|=k$

用树状图表示层次聚类

- 对于A中的每个节点a创建叶子节点
- 将节点聚合到相同的类别中
- 如果第k类包含了簇类B、C，创建一个新的节点index是k，子节点是B、C

### Agglomerative clustering(聚合型)

**Basic idea**

- 从n个簇类$c_i$开始，$c_i = {a_i}$
- 在每一步，将最近的两个簇类$c_i,c_j$替换成他们的并$c_i \cup c_j$
- 直到只剩下一个簇类

有两种方法评估最近

- complete Linkage

  对于$c_1,c_2 \subseteq M$，$D_{CL}(c_1,c_2) = max$ $D(x,y)$，其中$x \in c_1, y \in c_2$

  开销：$cost_{diam}^D(c_k) \leq O_d(logk)\cdot opt_k^{diam}(A)$

- single Linkage

  对于$c_1,c_2 \subseteq M$，$D_{CL}(c_1,c_2) = min$ $D(x,y)$，其中$x \in c_1, y \in c_2$

### Divisive clustering(分离型)

起始：所有点都在一个簇类

迭代：选择一个簇并分裂成两个子簇，直到所有的叶子节点只包含一个点

### HC based on Gonzalo algorithm

**Full-Farthest-First**

$D(A,C)$ $=$ $max_{a \in A}$ $min_{c \in C}$ $D(a,c)$, $C \subseteq A$

- 随机从A中挑选一个中心$c_1$，令$c^1 = {c_1}$

- 对于$i = 2,\cdots,|A|$

  记$R_i = max_{a \in A}$ $D(a, c^{i-1})$

  选取对应的$c_i$，令$c^i = c^{i-1} \cup {c_i}$

- 返回$c_i,\cdots, c_{|A|}$​和$R_2,\cdots, R_{|A|}$

对每个点定义等级：令$R=R_2$(最长的)，$L_0 = {c_1},L_j = \{c_i|R_i \in (\frac{R}{2^j}, \frac{R}{2^{j+1}}]\}$。记$L(x) = j$，如果$x \in L_j$​

每个点的父节点定义为$parent(c_i) = argmin\{D(x_i,y)|y \in \cup_j^{L(x_i-1)}L_j\}$

对于任意的$x \in A$，有$D(x,\cup_{j^{'}}^j L_{j^{'}}) \leq \frac{R}{2^j},D(x,parent(x)) \leq \frac{R}{2^{L(x)-1}}$

**层次聚类-k-center(A,D)**

1. 通过Full-Farthest-First计算$x_1,\cdots ,x_{|A|}$和$R_2,\cdots,R_{|A|}$

2. 记$R = R_2，L_0 = {x_1}, L(x_1) = 0$

3. $L_j = \{c_i|R_i \in (\frac{R}{2^j}, \frac{R}{2^{j+1}}]\}$ 并且记$L(x) = j$，如果$x \in L_j$​

4. 每个点的父节点定义为$parent(c_i) = argmin\{D(x_i,y)|y \in \cup_j^{L(x_i-1)}L_j\}$

5. 记$c_i = {x_i}$并且$H_{|A|} = {c_1,\cdots,c_{|A|}}$

6. for k = |A| to 1 do:

   令$x_p = parent(x_{k+1})$(注: $x_p \in c_p$)，令$c_p = c_p \cup c_k, H_k = \{c_i| i \in [k]\}$

7. 返回$H_1,\cdots,H_{|A|}$

## 感知机

找到一个超平面能够进行二分类

$f(x) = sign(w \cdot x +b)$

解决方案

- 线性规划算法
- 感知机算法
- 如果存在大的Margin下的基本可行解

**感知机学习算法**

1. 初始化$w,b$
2. 从训练集中选取$(x_i,y_i)$

3. 当存在$x_i$使得$y_i(w*x_i , b) \leq 0$时

​	$w \leftarrow w + \eta y_i x_i$

​	$b \leftarrow b + \eta y_i$

4. 重复2,3直到没有误分类点

**感知机收敛定理**：如果存在$w^*$对所有的$i$满足$(w^*)^Tx_iy_i \geq 1$，那么感知机算法可以在更新次数不超过$r^2||w^*||^2$的情况下，找到一个解$w$，其中$r = max||x_i||$​

$(w^*)^Tx_iy_i \geq 1$等价于点到超平面的距离不超过$$\frac{1}{||w||}$$

证明：

​	在每次更新时

​	$$w^Tw^* \rightarrow (w+x_iy_i)^Tw^* \geq w^Tw^* + 1$$

​	$$||w||^2 \rightarrow (w+x_iy_i)^T(w+x_iy_i) = ||w||^2 + 2x_i^Ty_iw +||x_i||^2 \leq ||w||^2 + r^2$$

​	(这里取小于号是因为仅当$x_i^Ty_iw \leq 0$时才会更新)

​	在$t$次循环后 $||w||^2\leq tr^2$

​	于是我们有 $$t \leq w^Tw^* \leq ||w||^2 \leq ||w||\sqrt{t}r$$

​	即$$t \leq ||w^*||^2r^2$$

## 支持向量机

寻找最优的超平面

### 线性可分

离超平面最近的称为支持向量，间隔\rho$是支持向量间的距离

最大化间隔分类，这暗示了只有支持向量起作用，其他的训练样本会被忽视。

**Linear SVM**

训练集$(x_i,y_i)$被一个间隔为$\rho$的超平面正确分割，对于每个样本有

$y_i(w^Tx_i+b) \geq \frac{\rho}{2}$

给定约束$()(w^*)^Tx_i+b)y_i \geq 1$

$r = \frac{1}{||w||}$，所以有$$\rho = 2r = \frac{2}{||w||}$$

问题变成了 寻找$w,b$使得$$\rho = \frac{2}{||w||}$$最大，并且$y_i(w^Tx_i+b) \geq 1$

重写上述问题 寻找$w,b$使得$$||w||^2$$最小，并且$y_i(w^Tx_i+b) \geq 1$

这是一个二次优化问题

然后展示如何求解这个问题

首先我们介绍拉格朗日乘子，将约束的目标函数转化为没有约束的目标函数

$$L(w,b,\alpha) = \frac{1}{2}||w||^2 - \sum_i^N\alpha(y_i(wx_i+b)-1)$$

$\theta(w) = maxL(w,b,\alpha)$

如果样本不满足约束条件，也就是$y_i(w^Tx_i+b) < 1$，则$\alpha_i \rightarrow +\infty,\theta(w) \rightarrow +\infty$

如果样本满足约束条件，那么$y_i(w^Tx_i+b) \geq 1$，则$\alpha_i = 0,\theta(w) = \frac{1}{2}||w||^2$

那么问题就变成了$min_{w,b}\theta(w) = min_{w,b}max_{\alpha_i\geq 0}L(w,b,\alpha) = p^*$

根据拉格朗日对偶性，等价于$max_{\alpha_i\geq 0}min_{w,b}L(w,b,\alpha) = d^*$

**原问题vs对偶问题**

Primal:

​	minimize $f(x)$

​	约束条件$g_i(x) \leq 0, h_i(x) = 0$

Lagrangian relaxation

​	$L(x,\lambda,D) = f(x) + \sum_i^n\lambda g_i(x)+\sum_i^pv_i h_i(x)$

​	约束条件$$\lambda_i \geq 0, v_i \geq 0$$

给定一个拉格朗日对偶问题

$g(\lambda,v) = inf_{x \in D}f(x) + \sum_i^m \lambda_i f_i(x) + \sum_i^l v_i h_i(x)$

$x \in D$表示$x$是原问题的解

$inf$代表最小值

对偶是一个$\lambda,v$的函数

对偶$g(\lambda,v)$是原问题的下界，因此最大化对偶问题可以接近原问题的最优解

KKT条件:

- 初始条件:$f_i(x) \leq 0, h_i(x) = 0$
- 对偶约束:$\lambda \geq 0$
- 对偶互补条件:$\lambda_i f_i(x) = 0$
- 对$x$的梯度为0

如果$x,\lambda,v$满足KKT条件，且原问题是一个凸优化问题，那么$x,\lambda,v$同时是原问题和对偶问题的最优解

回到线性SVM，问题可以描述为

寻找$w,b$使得$$||w||^2$$最小，并且$y_i(w^Tx_i+b) \geq 1$

对偶形式$$L(w,b,\alpha,\beta) = \frac{1}{2}||w||^2 + \sum_i^N\alpha(1 - y_i(wx_i+b))$$

使用KKT条件的第四条

$\frac{\partial(L(w,b,\alpha,\beta))}{\partial w} = w - \sum_i^n\alpha_i y_i x_i = 0$

$\frac{\partial(L(w,b,\alpha,\beta))}{\partial b} = \sum_i^n\alpha_i y_i = 0$

所以有$w = \sum_i^n\alpha_i y_i x_i, \sum_i^n\alpha_i y_i = 0$

带入上式

$$\begin{split} L(w,b,\alpha,\beta) &= \frac{1}{2}||w||^2 + \sum_i^n\alpha_i - \sum_i^n \alpha_iy_iw^Tx_i - b \sum_i^n \alpha_i y_i \\ &= \sum_i^n\alpha_i - \frac{1}{2}\sum_i^n\sum_j^n \alpha_i y_ix_i \alpha_j y_j x_j     \\ \end{split}$$

这样问题写成了

找到$a_1, \cdots, a_n$   使得$Q(a) = \sum_i^n\alpha_i - \frac{1}{2}\sum_i^n\sum_j^n \alpha_i y_ix_i \alpha_j y_j x_j$最大，并且有

$\sum a_iy_i = 0, a_i \geq0$其中至少有一个$a_i > 0$

给出了对偶问题$a_1,\cdots,a_n$的解后，原问题的解是

$w = \sum_i^n\alpha_i y_i x_i, b = y_k - \sum\alpha_i y_i x_i^Tx_k$  for any $\alpha_k > 0$

每一个非零的$\alpha_i$表示对应的$x_i$是一个支持向量

分类函数是$f(x) = \sum\alpha_i y_i x_i^T x + b$    (这说明了不用求出$w$)

这个函数取决于测试点$x$和支持向量$x_i$

序列最小优化算法SMO 核心思想一次只优化一个参数，固定其他的

### 软间隔

数据集不是线性可分时

问题描述：找到$w,b$，使得$\phi(w) = w^Tw +c \sum \xi_i$最小

并且对所有的$(x_i,y_i)$ 有$y_i(w^Tx_i+b) \geq 1 - \xi_i,\xi_i \geq 0$

对偶形式$$L(w,b,\alpha,\beta) = \frac{1}{2}||w||^2 + \sum_i^n\xi_i - \sum_i^N\alpha(1 - y_i(wx_i+b)) - \sum_i^n\beta_i\xi_i$$

根据TTK条件有$\frac{\partial L}{\partial w} = 0, \frac{\partial L}{\partial b} = 0, \frac{\partial L}{\partial \xi} = 0$

### kernel

#### 核函数k

- 初始化$\alpha = 0$(一个长度为n的向量)

- 执行下列过程

  对于所有$i$，令$\hat{y_i} = sgn \sum_j^k \alpha_i y_i k(x_j, x_i)$

  如果存在$i$使得$\hat{y_i} \neq y_i$，则$\alpha_i \leftarrow \alpha_{i+1}$

  否则返回$\alpha$

对于待预测的$x$，计算并输出$sgn\sum_j^k \alpha_j y_j k(x_j,x)$

#### kernel SVM

对偶问题描述为

找到$a_1, \cdots, a_n$   使得$Q(a) = \sum_i^n\alpha_i - \frac{1}{2}\sum_i^n\sum_j^n \alpha_i y_i \alpha_j y_j k(x_i,x_j)$最大，并且有$\sum a_iy_i = 0, a_i \geq 0$

分类函数是$f(x) = \sum\alpha_i y_i k(x_i^T x) + b$ 

#### 核函数的属性

给定样本$x_1,\cdots,x_n$和核函数k，核关联矩阵k = $(k_{ij})$，其中$k_{ij} = k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$

k是一个核矩阵，如果存在函数$\phi$使得$k_{ij} = \phi(x_i)^T\phi(x_j)$，那么k是半正定的

核函数是将线性空间映射到特征空间的函数。

如果$k_1,k_2$是合理的核函数，那么

- 对于任意的常数$c \geq 0$，$ck_1$​也是核函数
- $k_3(x,y) = f(x)f(y)k_1(x,y)$是合理的核函数
- $k_1 + k_2$是合理的核函数
- $k_1k_2$是合理的核函数

举例

- 线性：$k(x_i, x_j) = x_i^Tx_i$
- 多项式：$k(x_i,x_j) = (1+x_i^Tx_j)^p$
- 高斯：$k(x_i,x_j = e^{-\frac{||x_i-x_j||^2}{2\delta^2}})$