# chap 10

## 概率计数

考虑这样一个计数器问题：数据流中只有一种数据，也就是“按动计数器”操作。你的计数器可能随时被查询总共被按了多少次

这个问题十分平凡，一个精确算法如下：

- 维护一个数 𝑛，初始时为 0
- 计数器被按动时，令 𝑛←𝑛+1
- 查询时，返回 𝑛

但是实际上算法可以使用更少的空间。以下给出一个空间复杂度为 $o(⁡log⁡n)$ 的算法

### **Morris**

- 维护一个数 $X$，初始时为 $0$
- 计数器被按动时，以 $2^{-X}$ 的概率令 $X \leftarrow X+1$
- 查询时，返回 $n = 2^X - 1$​

需要的空间$O(loglogn)$

**证明：**考虑到 $x_0=0$ 即 $E\qty[2^{x_0}]=1$，而且 $$ \begin{split} \mathbb E\qty[2^{X_{n + 1}}] &= \sum_{j = 0}^{\infty}\mathbb E\qty[2^{X_{n + 1}} \mid X_n = j]\Pr[X_n = j]\\ &= \sum_{j = 0}^{\infty} \qty(2^{j + 1}\cdot \Pr\qty[X_{n + 1} = j + 1 \mid X_n = j] + 2^j\cdot\Pr\qty[X_{n + 1} = j \mid X_n = j])\Pr\qty[X_n = j]\\ &= \sum_{j = 0}^\infty \qty(2^{j + 1}\cdot 2^{-j} + 2^j\cdot (1 - 2^{-j}))\Pr\qty[X_n = j]\\ &= \sum_{j = 0}^\infty\qty(2^j + 1)\Pr\qty[X_n = j]\\ &= \sum_{j = 0}^\infty 2^j\Pr\qty[X_n = j] + \sum_{j = 0}^{\infty}\Pr\qty[X_n = j]\\ &= \mathbb E[2^{X_n}] + 1\\ &= n + 2 \\ \end{split} $$ 

$$E[\hat{n}] = E[2^{X_n-1}-1] = n$$

$$Var[n] = E[(\hat{n} - n)^2] = E[(2^{X_n}-1-n)^2] = E[2^{2X_n}] - (n+1)^2 = \frac{n^2}{2} - \frac{n}{2} < \frac{n^2}{2}$$

根据切比雪夫不等式$Pr[|\hat{n} - n| \geq \epsilon n] \leq \frac{Var[\hat{n}]}{\epsilon^2 n^2} < \frac{1}{2 \epsilon^2}$

可以看出来这样没什么用，当$\epsilon \geq 1$时概率才小于$\frac{1}{2}$，但这时误差很大了



### **Morris+**

- 独立运行$s$次Morris算法，设这$s$个Morris算法的输出分别是$\hat{n}_1,\hat{n}_2,\cdots,\hat{n}_s$
- 输出这些输出的算术平均$\hat{n} = \frac{1}{s} \sum_j^s\hat{n}_j$

重新考虑切比雪夫不等式$Pr[|\hat{n} - n| \geq \epsilon n] \leq \frac{Var[\hat{n}]}{\epsilon^2 n^2} < \frac{1}{2s \epsilon^2}$

对于任意的$\delta$，只需要取$s \geq \frac{1}{2\delta\epsilon^2}$，就能让$Pr[|\hat{n} - n| \geq \epsilon n] \leq \delta$



### **Morris++**

- 对于所有的$i \leq t$，如果第$i$个Morris+算法给出了正确的结果($\hat{n} - n| \leq \epsilon n$)，令$Y_i = 1$，否则记为$0$。要使得$Pr[Y_i = 1] > \frac{2}{3}$，也就是取$\delta = \frac{1}{3},s = \frac{3}{2\epsilon^2}$
- 输出$\hat{n}_1,\hat{n}_2,\cdots,\hat{n}_s$的中位数$\hat{n}$

记$Y = \sum_iY_i$为Morris算法中正确的数量，那么$E[Y] \geq \frac{2}{3}t$

因为输出的是中位数$\hat{n}$，只要$Y \geq \frac{1}{2}t$，就能给出理想的结果

$Pr[\hat{n} = bad] \leq Pr[Y \leq \frac{t}{2}] < Pr[Y-E[Y]<-\frac{t}{6}] \leq e^{-2(\frac{1}{6})^2t} \leq \delta$

当$t \geq 18ln(\frac{1}{\delta})$时，就能得到需要的结果

所需要的空间

- 算法需要的s,t   ($O(\frac{1}{\epsilon^2}ln(\frac{1}{\delta}))$)

- 我们接下来证明：如果调用了 $s^*$ 次 Morris 算法，那么以至少 $\delta^*$ 的概率，在这 𝑛 次计数过程中每一个数都不超过 $\log\qty(\frac{s^*n}{\delta^*})$

  假设某个 Morris 算法在过程中达到了 $X = \log(\frac{s^*n}{\delta^*})$，那么它再增加一次的概率是 $\frac 1{2^X}\leq\frac{\delta^*}{s^*n}$。因为 $X$ 总共只有 $n$ 个机会增加，所以 $X$ 在算法结束的时候有至多 $\frac n{2^X}\leq\frac{\delta^*}{s^*}$ 的概率增加。总共有 $s^*$ 个 Morris 算法，至多有 $s^*$ 个数达到了 $\log\qty(\frac{s^*n}{\delta^*})$ 随时准备突破。所以在算法结束的时候有至多 $\delta^*$ 的概率某个 Morris 算法的 $X$ 超过 $\log\qty(\frac{s^*n}{\delta^*})$ 了。这就表明有至少 $1 - \delta^*$ 的概率所有到达过临界值 $\log\qty(\frac{s^*n}{\delta^*})$ 的 𝑋 都不会再增长了，也就表明有至少 1−𝛿∗ 的概率所有 Morris 算法中的 𝑋 都不超过 $\log(\frac{s^*n}{\delta^*})$

  这就表明 Morris++ 算法以至少 $1 - \delta^*$ 的概率空间复杂度为 $$ \order{st\log\log(\frac{stn}{\delta^*})} = \order{\qty(\frac 1\epsilon)^2\ln\frac 1\delta\log\log\frac {n\log\frac1\delta}{\epsilon^2\delta^*}} $$

## 蓄水池抽样

有一个数据流，数据在不停地流过——你只能看到每个数据一次并且不能将它们全部存储下来。你在任意时刻都可能被要求：“将刚刚你看到的所有数中均匀随机抽取一个给我。”

假设看到的数依次为 $\{a_i\}_{i = 1, 2, \cdots,\infty}$。的思想，可以实现如下地算法：

- 维护变量 $s$，初始时值未定义
- 当看到数据 $a_m$ 的时候，掷骰子，以 $\frac{1}{m}$ 的概率令 $s \leftarrow a_m$，以 $1 - \frac{1}{m}$ 的概率保持 $s$ 不变
- 查询时，直接输出 $s$

这个算法的正确性比较显然。

当数据流中流过 $m$ 个数的时候，如果 $s = a_i$，那么第 $i$ 次掷骰子掷得了 $\frac{1}{i}$ 将 $a_i$ 保留下来，而在第 $i+1,i+2,\cdots,m$ 的时候掷得了 $1−\frac{1}{i+1},1-\frac{1}{i+2},⋯,1−\frac{1}{m}$ 而没有将 $a_i$ 扔掉。

这些事件都是随机的，所以 $$ \Pr\qty[s_m = a_i] = \frac 1i\cdot \frac i{i + 1}\cdot \frac{i + 1}{i + 2}\cdots\frac {m - 1}m = \frac 1m $$ 对于任意的 $i = 2,3,\cdots,m$ 均成立，也就是说 $a_1,a_2,\cdots,a_m$ 能等概率地在第 $m$ 次出现

分析一下空间复杂度。我们需要存储被抽样的数与当前经过了多少数。假设所有数都不超过 𝑛，那么我们的空间复杂度就是 $\order{log⁡n+log⁡m}$

## 估计不同元素个数

需要在任意时刻估计当前数据流中不同元素的个数，并且用尽可能少的事件。设数据流是 $\{a_i\}_{i = 1,2,\cdots}$ 而且所有数都是整数，你需要在逐一读取这些数据的时候随时准备好回答 不同元素个数

设这个数据流中数的上界是 𝑛，在读取完前 𝑚 个数的时候被要求作答。如果需要求出精确解：

1. 维护一个长度为 $n$ 的数组 ，如果第$i$个数看见了就置为$1$(space: $n$ bits)

2. 将全部数据流存在内存中(space: $mlog(n)$ bits)

   需要$min\{n,mlog(n)\}$比特内存

我们想要一个空间复杂度是 $\order {log⁡(nm)}$ 的近似算法，也就是需要对任意的参数 $\epsilon,\delta \in (0,1)$，求出一个 $\hat{t}$ 满足 $Pr\qty[|\hat{t}-t| > \epsilon t] < \delta$，其中 $t$ 是数据流中不同元素的个数

### FM算法

- 选择一个随机函数$h:[n] \rightarrow [0,1]$

- 保留一个计数器$X:min_{i\in stream}h(i)$
- 输出$\hat{t} = \frac{1}{x}-1$

断言1：$E[X] = \frac{1}{t+1}$

证明：

​	$\begin{split} E[X] &= \int_0^{\infty}Pr[X \geq \lambda]d\lambda \\ &= \int_0^{\infty}Pr[i \in stream,h(i)\geq\lambda] \\ &= \int_0^{\infty}\Pi_r^t Pr[h(i_r) \geq \lambda]d\lambda \\ &= \int_0^1(1-\lambda)^td\lambda \\ &= \frac{1}{t+1} \end{split}$

断言2：$E[X^2] = \frac{2}{(t+1)(t+2)}$

证明：

​	$\begin{split} E[X^2] &=\int_0^{\infty}Pr[X^2\geq\lambda]\lambda \\ &= \int_0^{\infty}Pr[X\geq\sqrt{\lambda}]d\lambda \\ &=\int_0^{\infty}(1-\sqrt{\lambda})^td\lambda \\ &= 2\int_0^1u^t(1-u)du \\ &=\frac{2}{(t+1)(t+2)}   \end{split}$

因此$Var[X] = E[X^2] - (E[X])^2 = \frac{t}{(t+1)^2(t+2)} < \frac{1}{(t+1)^2}$



### FM+算法

- 独立运行$s = \frac{25}{\epsilon^2\delta}$次FM算法，并且得到结果$X_1,\cdots,X_s$

- 令$Z = \frac{1}{s}\sum_i^sX_i$，输出$\hat{t} = \frac{1}{z}-1$

断言3：$Pr[|Z - \frac{1}{t+1}| > \frac{\delta}{5(t+1)}] < \delta$

证明：

​	$\begin{split} E[Z] &= \frac{1}{t+1} \\ \end{split}$ 

​	$Var[Z] < \frac{1}{s(t+1)^2}$

​	由于切比雪夫不等式

​	$Pr[|Z-\frac{1}{t+1}| > \frac{\epsilon}{5(t+1)}] < \frac{25(t+1)^2}{\epsilon^2}\frac{1}{s(t+1)^2} < \delta$

断言4：$Pr[|(\frac{1}{z}-t| > \epsilon t] < \delta$，if $\epsilon < \frac{1}{2}$

证明：

​	如果$t = 0$，结果是显然的

​	如果$t \geq 1$，根据断言3，有大于$1-\delta$的概率满足

​	$\frac{1-\frac{\epsilon}{5}}{t+1} < Z < \frac{1+\frac{\epsilon}5{}}{t+1}$

​	$\rightarrow \frac{t+1}{1+\frac{\epsilon}{5}} < \frac{1}{Z} < \frac{t+1}{1-\frac{\epsilon}{5}}$

​	$\rightarrow \frac{t-\frac{\epsilon}{5}}{1+\frac{\epsilon}{5}} < \frac{1}{Z} - 1 < \frac{t+\frac{\epsilon}{5}}{1-\frac{\epsilon}{5}}$

​	$\rightarrow \frac{1}{Z} - 1 > \frac{t-\frac{\epsilon}{5}}{1+\frac{\epsilon}{5}} = t -\frac{2\epsilon t}{5} + \frac{\epsilon^2}{25} > t - \epsilon t$

​	同理$\frac{1}{Z} < t + \epsilon t$ (需要利用$\epsilon \leq \frac{1}{2}$)

​	所以有$|\frac{1}{Z} - 1 - t| < \epsilon t$



### FM++算法

- 独立运行$q = 18ln(\frac{1}{\delta})$次FM+算法，其中$\delta_{FM+} = \frac{1}{3}$。得到结果$\hat{t}_1,\cdots,\hat{t}_q$
- 输出得到$\hat{t}_1,\cdots,\hat{t}_q$的中位数$\hat{t}$

断言5：$Pr[|\hat{t}-t| > \epsilon t] \leq \delta$

​	证明和Morris++时一样



# Frequent Items/Heavy hitters

## 找到频繁项

输入：一个整数数据流$i_1,\cdots,i_m$和一个整数$k \geq 1$

问题：找到所有出现频率$f_i > \frac{m}{k+1}$的整数$i$

**Misra-Gries 算法**

用数组$A$记录$k$项和他们的次数

1. 如果下一个$x$是计数器$k$项中的，令对应计数器加一
2. 如果有计数器为空，将$x$放入$A$并计数为1
3. 否则令全部$k$项计数器减一，等于0的项移出计数器

输出所有$A$中的$k$项

定理：数据流中任意频率$f_i > \frac{m}{k+1}$的项一定在$A$中

再遍历一遍判断是否存频率真的大于$\frac{m}{k+1}$

为了证明这个命题，我们定义一个辅助变量 $\hat{f}_i$：如果 $i$ 在算法结束的时候在 $A$ 中，那么 $\hat{f}_i$ 就是它对应计数器的值；如果 $i$ 不在，那么 $\hat{f}_i = 0$。

定理：$f_i - \frac{m}{k+1} \leq \hat{f}_i \leq f_i$对任意的$i$成立

证明：

​	将算法的描述修改为等价形式

​	1. 对每个数 $i = 1,2,\cdots,n$ 都维护一个计数器 $\hat{f}_i$，初始时$\hat{f} \leftarrow \hat{0}$ 

​	2. 当数据流入第$i$个数时

​		如果$\hat{f}_i > 0$，那么$\hat{f}_i \leftarrow \hat{f}_i+1$

​		如果$\hat{f}_i = 0$且当前计数器$\hat{f}$中的整数不足$k$个，$\hat{f}_i = 1$

​		否则$\hat{f}$中的正数全部自减$1$

显然满足$\hat{f}_i \leq f_i$

设 $\alpha_i = f_i - \hat{f}_i$。数据流初始化没有流入任何数的时候，有 $\alpha = 0$。当流入第 $j$ 个数的时候

- 如果 $i_j = i$，且当前步骤中 $\hat{f}_i$ 自增了 1，那么这对应上述的第 1 种或者第 2 种自增情况，$\alpha_i$ 不会变化
- 如果 $i_j=i$，且当前步骤中 $\hat{f}_i$ 没有变化，那么这对应上述的第 3 种全部自减情况，$\alpha_i$ 增加 1
- 如果 $i_j \neq i$，且当前步骤中 $\hat{f}_i$ 没有变化，那么很高兴地 $\alpha_i$ 也不会变化
- 如果 $i_j \neq i$，且当前步骤中 $\hat{f}_i$ 自减了 1，那么这一定是上述的第 3 种全部自减情况导致的，$\alpha_i$ 增加 

设全部自减发生了$l$次，每次$\alpha_i$增加都是全部自减导致的，所以$\alpha_i \leq l$

全部自减发生了 $l$ 次，数据流种总共流入了 $m$ 个数，所以自增发生了 𝑚−𝑙 次，自增总共让 $\sum\hat{f}_i$ 增加了 $m-l$，而全部自减让 $\sum\hat{f}_i$ 减少了 $kl$

因为$\sum\hat{f}_i \geq 0$，所以$\sum\hat{f}_i = m-(k+1)l \geq 0$，也就是$l \leq \frac{m}{(k+1)l}$

所以$f_i - \hat{f}_i = \alpha_i \leq l \leq \frac{m}{(k+1)l}$

空间复杂度$O(k(logn+logm))$比特

## Turnstile Streaming Model

我们在流入 $i_j \in \{1,2,\cdots,n\}$ 的时候，同时流入一个辅助变量$\Delta_j \in R$。流入 $(i_j, \Delta_j)$ 意味着令 $x_{i_j} \leftarrow x_{i_j} + \Delta_j$

Turnstile Streaming Model 不对 $\Delta_j$ 进行任何限制，只要 $\Delta_j \in R$ 就行。

## 两个问题

从本节开始，我们默认使用 Strict Turnstile Streaming Model，而非传统的数据流模型

$(k,l_1)$ 单点查询问题指：任意时刻询问任意的 $i \in \{1,2,\cdots,n\}$，需要输出 $\widetilde{x}_i = x_i \pm \frac{1}{k}||x||_1$

$(k,l_1)$ 众数查询问题指：任意时刻进行询问，需要输出集合 $L$ 满足：

- $|L|=\order{k}$
- 如果 $|x_i| > \frac{1}{k}||x||_1$，那么 $i \in L$

> Misra Gries 算法就是在传统数据流模型下解决以上问题的一个算法

**引理：**设 𝐴 是一个 $(3k,l_1)$ 单点查询问题的算法，其失败概率不超过 $\frac{\delta}{n}$，且消耗空间为 $s$ 比特。那么可以构造出一种解决 $(k,l_1)$ 众数查询问题的算法 $A^{'}$，失败概率不超过 $\delta$，消耗 $s+\order{klog⁡n}$ 比特

在 $A$ 算法的基础上我们这样构造 $A^{'}$：查询时，遍历 $i=1,2,\cdots,n$，使用 𝐴 算法依次单点查询这 $n$ 个点的值，同时记住前 $3k$ 大的数并输出

我们证明所有出现次数不低于 $\frac{1}{k}||x||_1$ 的数都被输出了。因为一个单点查询失败的概率不超过 $\delta n$，所以所有单点查询失败至少一次的概率至多为 $\delta$，这些单点查询全部成功的概率至少为 $1-\delta$

- 如果 $x_i > \frac{1}{k}||x||_1$，则$\hat{x}_i > \frac{1}{k}||x||_1 - \frac{1}{3k}||x||_1 = \frac{2}{3k}||x||_1$
- 如果 $x_i \leq \frac{1}{3k}||x||_1$，则$\hat{x}_i \leq \frac{1}{3k}||x||_1 + \frac{1}{3k}||x||_1 = \frac{2}{3k}||x||_1$

这说明有不超过$3k$个$x_i \geq \frac{1}{3k}||x||_1$，给出的集合一定包含了所有$x_i \geq \frac{1}{k}||x||_1$

需要的空间也就是$s+3k\order{logn} = s+\order{klog⁡n}$

## Count Min Sketch

针对 Strict Turnstile Streaming Model 来给出一个 $(k,l_1)$ 单点查询算法。回忆一下 Strict Turnstile Streaming Model，该模型要求输入满足任意时刻 $x$ 中的所有元素都是非负

先给出算法。设 $w,d$ 是两个参数，我们后面再对其选择：

1. 选择 $d$ 个 2 元独立随机哈希函数 $h_1,h_2,\cdots,h_d$，它们将 ${1,2,\cdots,n}$ 均匀地散列到 ${1,2,\cdots,w}$ 上
2. 维护一个二维数组 $C$，初始化 $C_{i,j} = 0$
3. 当数据流中流入 $e_t = (i_t,\Delta_t)$ 的时候，遍历每个哈希函数 $h_1,h_2,\cdots,h_d$。当遍历到函数 $h_l$ 的时候，更新 $C_{l,h_l(i_t)} \leftarrow C_{l,h_l(i_t)} + \Delta_t$
4. 记 $\hat{x} = min_lC_{l,h_l(i)}$，作为查询 $x_i$ 时的输出结果

引理1：

​	考虑 Strict Turnstile Streaming Model，令$d = \Omega(log(\frac{1}{\delta}))$并且$w > 2k$。则对于给定的$i$有$x_i \leq \hat{x}_i$并且$Pr[\hat{x}_i \geq x_i + \frac{||x||_1}{k}] \leq \delta$

证明：

​	给定$l$和$i$，$h_l(i)$是一个桶，将$i$映射到$Z_l = C[l,h_l(i)]$

​	$\begin{split} E[Z_l] &= x_i + \sum_{i^{'} \neq i}Pr[h_l(i^{'}) = h_l(i)]x_{i^{'}}\\ &=x_i +\sum_{i^{'} \neq i}\frac{x_i}{w}\\ &< x_i +\frac{1}{2k}||X||_1 \\ \end{split}$

​	因为考虑的是Strict Turnstile Streaming Model，$Z_l - x_i$是非负的。

​	根据马尔可夫不等式$Pr[Z_l - x_i > \frac{||X||_1}{k}] < \frac{1}{2}$

​	由于哈希函数相互独立

​	$Pr[min_lZ_l \geq x_i + \epsilon||X||_1] < (\frac{1}{2})^d < \delta$

​	如果我们选择$d = \Omega(log(n)), w=3k$，那将有$1-\frac{1}{n}$的概率满足$\hat{x}_i \leq x_i +\frac{||X||_1}{k}$

​	空间复杂度$\order{klognlogm}$比特

## Count Sketch

Count Min Sketch 算法中空间复杂度大约是关于 𝑘 的线性关系，还是可以使用类似 Count Min Sketch 的算法与分析来求解 𝑙2 范数中的问题

1. 选择 $d$ 个 2 元独立随机哈希函数 $h_1,h_2,\cdots,h_d$，它们将 ${1,2,\cdots,n}$ 均匀地散列到 ${1,2,\cdots,w}$ 上，再选择 $d$ 个 2 元独立随机哈希函数 $g_1,g_2,\cdots,g_d$，它们将 ${1,2,\cdots,n}$ 均匀地散列到 $\{-1,1\}$ 上
2. 维护一个二维数组 $C$，初始化 $C_{i,j} = 0$
3. 当数据流中流入 $e_t = (i_t,\Delta_t)$ 的时候，遍历每个哈希函数 $h_1,h_2,\cdots,h_d$。当遍历到函数 $h_l$ 的时候，更新 $C_{l,h_l(i_t)} \leftarrow C_{l,h_l(i_t)} + g_l(i_t)\Delta_t$
4. 记 $\hat{x} = median\{g_l(i)C_{l,h_l(i)} \}$，作为查询 $x_i$ 时的输出结果

仿照上一问，但是$E[Z_t] = X_i$

$Var[Z_t] = \sum_{i^{'} \neq i}x_{i^{'}}^2E[Y_{i^{'}}^2] < \frac{||X||_2^2}{w}$

然后先用切比雪夫不等式得到

$Pr[|Z_t-x_i|\geq \frac{||X||_2}{k}] \leq \frac{k^2}{w} < \frac{1}{3}$

再利用Morris++，FM++中关于中位数的方法解决