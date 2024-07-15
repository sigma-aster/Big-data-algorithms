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