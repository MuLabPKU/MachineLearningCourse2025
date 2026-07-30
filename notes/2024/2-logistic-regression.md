---
title: 24秋机器学习笔记-02-逻辑回归
date: 2024-09-18 23:09:12
tags:
  - 本科课程
  - 机器学习
categories: [笔记, 本科课程, 机器学习]
---

> 本文主要涉及逻辑回归（Logistic Regression）及相关拓展。
> [一键回城](/note-ml2024fall/)。

## 问题描述

虽然叫“回归”，但是实际是用于处理**二分类**（binary classification）问题的。记号约定与之前类似：$x \in  \mathbb{R}^d$，但是 $y \in \{0,1\}$，$f(x) = w^Tx + b$ 不变。

不过，如果我们需要 hard prediction（即 $y_i = 0$ 或 $1$），怎么从这个 $f(x) \in \mathbb{R}$ 得到？更进一步地，如果我们需要的是一个所谓的 $y_i = 1$ 的**概率**（soft prediction）呢？

或者说，可以这样描述我们的问题：

1. $P(y = 1 \mid x) \in [0,1]$ 是有界的，但 $f(x) \in \mathbb{R}$ 无界；
2. 并没有一个针对 $P(y = 1\mid x)$ 的 ground truth，我们只有硬分类数据 $\{(x_i, y_i)\}$ 而非 $\{(x_i, P(y_i = 1|x_i))\}$。

为了解决第一个问题，我们引入 Sigmoid 函数：

$$
\sigma(x) = \frac{1}{1 + \exp(-x)}
$$

这个函数的性质特别好，其在 $\mathbb{R}$ 上单调，并且值域是 $(0,1)$，所以我们可以用 $\sigma(f(x))$ 来刻画这个概率 $P(y = 1\mid x)$。如果想要一个硬分类，那就设一个阈值 $k$，对于预测结果 $\ge k$ 的情况，预测为 $1$；否则预测为 $0$。注意到此时 $w^Tx + b$ 就是所谓的分隔超平面（seperating hyperplane）

## 最大似然估计（MLE）

为了解决第二个问题，我们使用**最大似然估计法**（Maximum Likelihood Estimation, MLE），其哲学在于**找寻使得你的概率模型在观测数据上的似然达到最大化的参数**。

对于逻辑回归，单个数据点的似然即为

$$
P(y = y_i \mid x = x_i) = \begin{cases} y_i = 1, &P(y = 1\mid x=x_i) = \sigma(f(x_i)) \\ y_i = 0, &P(y = 0\mid x = x_i) = 1 - \sigma(f(x_i)) = \sigma(-f(x_i))  \end{cases}
$$

对于所有的数据点，乘起来即可：

$$
\prod_{i \in [n]} P(y = y_i \mid x = x_i) = \prod_{i \in [n]}\left[ \sigma(f(x_i))^{y_i}(1 - \sigma(f(x_i)))^{1 - y_i} \right] 
$$

最大化之等价于最大化其对数（对数似然，log-likelihood），即

$$
\max_{w,b} \sum_{i \in [n]} \left[ y_i \log \sigma(f(x_i)) + (1 - y_i)\log(1 - \sigma(f(x_i))) \right] 
$$

（考虑到 $\sigma(z) \in (0,1)$，连乘起来可能出现数值上无穷小，所以不妨取对数）

最大化对数似然等价于

$$
\min_{w,b} \left\{ -\sum_{i \in [n]} \left[ y_i \log \sigma(f(x_i)) + (1 - y_i)\log(1 - \sigma(f(x_i))) \right] \right\} 
$$

括号里的东西称为交叉熵损失（Cross Entropy Loss, CELoss）

### 有关熵的讨论

熵（entropy）可用于描述“混乱程度”，对于随机变量 $Y$，其熵被定义为

$$
\begin{aligned}
H(Y) &= -\sum_y P(y) \log P(y) \\
&= \sum_y P(y) \log \frac{1}{P(y)}
\end{aligned}
$$

“各个事件的等价程度越高，熵就越高”。

$P(y\mid x_i)$ 的条件熵：

$$
\begin{aligned}
&-\sum_{\hat{y_i} \in  \{0,1\}} P(y = \hat{y_i}\mid x_i) \log P(y = \hat{y_i}\mid x_i) \\
\end{aligned}
$$

交叉熵涉及两个分布 $P$ 与 $Q$。其中 $P$ 为实际分布，$Q$ 为模型预测的分布。其度量的是用分布 $Q$ 编码分布 $P$ 所需的平均信息量，表达式为

$$
H(P,Q) = -\sum_{x} p(x) \log q(x)
$$

在这个例子中，就是

$$
-(y_i \log \sigma(f(x_i)) + (1 - y_i)\log(1 - \sigma(f(x_i))))
$$

## 线性可分与过拟合

跟之前类似地，设 $\hat{w} = \begin{bmatrix} w \\ b \\\end{bmatrix}$，$\hat{x} = \begin{bmatrix} x \\ 1 \\\end{bmatrix} \in  \mathbb{R}^{d+1}$。

$$
\begin{aligned}
L(\hat{W}) &= -\sum_{i \in [n]}\left( y_i \log \frac{1}{1+e^{-\hat{w}^T \hat{x_i}}} + (1 - y_i) \log\frac{1}{1 + e^{\hat{w}^T \hat{x_i}}} \right)  \\
&= - \sum_{i \in  [n]}\left( -y_i \log (1 + e^{-\hat{w}^T \hat{x_i}}) + (y_i - 1) \log(1 + e^{\hat{w}^T \hat{x_i}}) \right) \\
&= -\sum_{i \in [n]} \left[ y_i \hat{w}^T \hat{x_i}  - \log(1 + e^{\hat{w}^T \hat{x_i}}) \right] 
\end{aligned}
$$

这个形式方便我们进行求导：

$$
\begin{aligned}
\frac{\partial L(\hat{w})}{\partial \hat{w}} &= - \sum_{i \in [n]}\left( y_i \hat{x_i} - \frac{\hat{x_i} e^{\hat{w}^T \hat{x_i}}}{1 + e^{\hat{w}^T \hat{x_i}}} \right)  \\
&= -\sum_{i \in [n]} \hat{x_i} (y_i - \sigma(\hat{w}^T \hat{x_i}))\\
&= -\sum_{i \in [n]} \hat{x_i} (y_i - P(y = 1\mid x_i))
\end{aligned}
$$

如果我们希望 $\displaystyle \frac{\partial L}{\partial \hat{w}} = 0$，则说明 $\forall i \in [n]$ 有 $y_i - \sigma(\hat{w}^T \hat{x_i}) = 0$。但是这种情况通常是不可能出现的，除非训练数据**线性可分**（linearly separatable）

事实上这种情况通常并不是我们希望的。因为对于分隔超平面 $w^Tx + b = 0$，如果同时给 $w$ 和 $b$ 乘上系数 $k > 0$，则超平面在几何上是不变的（而且数据点到超平面的距离也是不变的），但是这会影响模型的预测值——因为 $w^Tx + b$ 处在 $e$ 的指数位上，所以这会使得 $\sigma(w^Tx+b)\to 0$ 或 $1$，即往使损失函数减小的方向上持续更新，进而导致 $k \to \infty$，模型越来越 sharp，造成**过拟合**（overfitting）。

如何避免这种情况？加一个 L2-Norm 就好了。

> 番外：GD 的物理含义
>
> $$
> \hat{w} \gets \hat{w} + \alpha \sum_{i \in [n]} [y_i - P(y_i = 1\mid x_i)] \hat{x_i}
> $$
> 
> 使得 $\hat{w}$ 往 $\hat{x_i}$ 的方向走

## 交叉熵损失的凸性

考察 Hessian 矩阵的半正定性。

$$
\begin{aligned}
\frac{\partial ^2 L(\hat{w})}{\partial \hat{w} \partial \hat{w}^T} &\in \mathbb{R}^{(d+1)\times (d+1)} \\
&= \sum_{i \in [n]} \frac{\partial \left( \frac{1}{1 + \exp(-\hat{w}^T \hat{x_i})} \hat{x_i} \right) }{\partial \hat{w}^T}\\
&= \sum_{i \in [n]} \frac{1}{(1 + \exp(-\hat{w}^T \hat{x_i}))^{2}} \cdot  \exp(-\hat{w}^T \hat{x_i})\cdot \hat{x_i} \cdot \hat{x_i}^T\\
&= \sum_{i \in [n]} \frac{1}{1 + \exp(-\hat{w}^T \hat{x_i})} \cdot  \frac{\exp(-\hat{w}^T \hat{x_i})}{1 + \exp(-\hat{w}^T \hat{x_i})}\cdot \hat{x_i} \cdot \hat{x_i}^T\\
&= \sum_{i \in [n]} P(y =1\mid x_i)\cdot P(y=0\mid x_i) \cdot \color{red}{\hat{x_i}\cdot \hat{x_i}^T} 
\end{aligned}
$$

首先注意到 $P(y = 1\mid x_i)\cdot P(y=0\mid x_i)\ge 0$，且 $\sum \hat{x_i} \hat{x_i}^T$ 为半正定矩阵，所以交叉熵损失函数为凸函数。

> 为什么 $\sum \hat{x_i} \hat{x_i}^T$ 为半正定矩阵？
>
> 从定义出发就行了：$\forall v \ne 0$，有 $v^T \hat{x_i} \hat{x_i}^T v = \left\| \hat{x_i}^T v \right\|^2 \ge 0$

## 平方损失函数？

即

$$
\min_{\hat{w}} \sum_{i \in [n]}(y_i - \hat{w}^T \hat{x_i})^{2}
$$

答案：不行。分析如下：

- 分类标签的 $y_i$ **没有数值上的意义**，$\{0,1\}$ 换成 $\{1,-1\}$ 是一样的。比如说 $y_i = 1$，$f(x_i) = 0.8$ 或 $1.2$，损失函数值都是一样的，失去了概率意义。
- $f(x_i) \in \mathbb{R}$，值域与 $\{0,1\}$ 是不匹配的。
- 当 $y_i = 0$ 且 $f(x_i) = 1$ 的时候，损失函数值仅仅为 $1$，反观如果使用 CELoss，$\sigma(f(x_i)) \to  1$ 的时候，造成的损失为 $-(1 - y_i) \log \sigma(-f(x_i)) \to +\infty$。
- 对离群值不健壮（not robust to **outlier**）：对于一个和分隔超平面很远的 outlier，用平方损失的话会造成把分隔平面往 outlier 方向拉的情况。

> 其实说了半天，最终道理还是因为分类标签本身是不具有数值意义的，所以不能用平方损失。如果使用平方损失的话会造成很多问题。

## Softmax Regression

该模型用于处理多分类问题（multiclass classification）。

记号约定：

- $y\in\{1,2,3, \cdots ,K\}$，代表在做 $K$ 分类，**只是 index 而已**，绝对大小并不重要，**所以不可以使用 squared loss 来做**。
- 有 K 个 classifier，$f_k(x) = w_k^Tx + b_k$，$k\in [K]$。

每个 classifier 输出的是对相应类别的一个打分，如何归约到概率上？

Softmax：

$$
P(y=k|x) = \frac{\exp(w_k^Tx + b_k )}{\sum_{j\in [K]}\exp(w_j^Tx + b_j)}
$$

性质：

1. It is a probability distribution, since $\displaystyle \sum_{k \in  [K]} P(y=k|x) = 1$，且 $P(y=k|x)\ge 0$（指数函数的性质）
2. 若 $w_k^Tx + b_k \gg w_j^Tx + b_j,\forall j\ne k$，则 $P(y=k|x) \approx 1$，且 $P(y=j|x) \approx 0$
   指数函数的**放大效应**。

考虑使用 MLE 来优化，写出 log-likelihood：

$$
\sum_{i \in [n]} \log \frac{\exp(w_{y_i}^Tx_i + b_{y_i})}{\sum_{j \in [K]}\exp(w_j^T x_i + b_j)}
$$

实际上，不一定要 $k$ 个线性分类器，其实可以用一整个神经网络，然后最后使用 softmax。

> Logistic Regression 和 Softmax Regression 的区别与联系：
>
> $K=2$ 的时候，假设 $1$ 为正类，$2$ 为负类，则
> $$
> \begin{aligned}
> P(y=1|x) &= \displaystyle \frac{\exp(w_1^Tx + > b_1)}{\exp(w_1^Tx + b_1) + \exp(w_2^Tx + b_2)} \\
> &= \frac{1}{1 + \exp((w_2-w_1)^Tx + (b_2-b_1))}\\
> &\text{let }b = b_1-b_2,w= w_1-w_2,\\
> &= \frac{1}{1+\exp(w^Tx+b)} = \sigma(w^Tx+b)
> \end{aligned}
> $$
> 等价于逻辑回归。

## 用最大似然估计解释线性回归

回归到了回归问题（逻辑回归和 softmax 回归都不是回归而是分类问题）。

注意到一开始推导线性回归的时候用的是 ERM，而非 MLE，现在考虑利用 MLE 来推导线性回归。

### Gaussian Distribution

一维高斯分布：$x \sim \mathcal{N}(\mu, \sigma^2)$，其中 $\mu$ 为 mean（均值），$\sigma^2$ 为 variance（方差）

概率密度函数（PDF）为（需要记忆）：

$$
P(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left( - \frac{(x-\mu)^2}{2 \sigma^2} \right) 
$$

是一个钟形曲线。优秀性质：$[\mu-2\sigma,\mu+2\sigma]$ 内，概率占 $95\%$。

More on *PRML* 2.3。

中心极限定理。

### 正式推导

由中心极限定理，假设 $y = w^Tx + b + \varepsilon$，其中 $\varepsilon$ 为噪声项，$\varepsilon \sim \mathcal{N}(0, \sigma^2)$。

则在如上假设中，$y$ 的条件分布（概率密度函数）为 $P(y|x;w,b,\sigma^2) = \mathcal{N}(w^Tx + b, \sigma^2)$（$w,b$ 为参数，$\sigma$ 为超参数，用分号隔开）

最大化 log-likelihood：

$$
\begin{aligned}
&\sum_{i \in [n]} \log \frac{1}{\sqrt{2\pi \sigma^2}}\exp\left( - \frac{(y_i - (w^Tx_i + b))^{2}}{2\sigma^{2}} \right)  \\
=&\sum_{i \in [n]}\left[ -\frac{1}{2}\log (2\pi \sigma^{2}) - \frac{1}{2\sigma^{2}}\left( y_i - (w^T x_i + b) \right)^{2}  \right]    \\
\end{aligned}
$$

只和 $\sigma$ 有关的项都可以提出来，这就等价于优化

$$
\min_{w,b} (y_i - w^Tx_i - b)^2
$$

## Maximum A Posteriori (MAP) 最大后验框架推导岭回归

之前，我们将 $w,b$ 当成未知的**固定**参数。不过在贝叶斯学派的视角中，一切都是随机变量（不确定），the world is uncertain and even the parameters $w,b$ are **random variables**。

之前的想法中，$y = w^Tx + b+ \varepsilon$，只有 $\varepsilon$ 是不确定的，但在贝叶斯学派中，$w,b$ 也是不确定的，会假设其有一个先验分布。

依旧令 $\hat{W} = \begin{bmatrix} w \\ b \\\end{bmatrix}$，假设其**先验分布**（prior dist.）为 $\hat{W}\sim \mathcal{N}(0, \sigma_w^2 I)$,其中 $\sigma_w^2 I \in \mathbb{R}^{(d+1)\times (d+1)}$ 为协方差矩阵，假设其为对角阵（$\hat{W}$ 的每一维独立）。

依旧假设 $y = \hat{W}^T \hat{x} + \varepsilon, \varepsilon \sim \mathcal{N}(0,\sigma^2)$，则

$$
\begin{aligned}
P(y|x,\hat{W}) = \mathcal{N}(\hat{W}^T\hat{x}, \sigma^2) \\
\end{aligned}
$$

现在我们有一堆观测数据，希望求得一个 $\hat{W}$ 的后验分布。

$$
\begin{aligned}
P(\hat{W} | x,y) &= \frac{P(y|x,\hat{W}) \cdot P(\hat{W}|x)}{P(y|x)} \qquad \text{Bayes formula}\\
& \text{note that }P(\hat{W}|x) = P(\hat{W})\\
& \text{and that }P(y|x)\text{ is unrelevant to } \hat{W}\text{, let it be }Z\\
&= \frac{1}{Z} \left( \frac{1}{\sqrt{2\pi \sigma^{2}}} \right)^n \exp\left( - \frac{1}{2 \sigma^{2}} \sum_{i \in [n]} (y_i - \hat{W}^T \hat{x_i})^{2} \right)  \cdot  \left( \frac{1}{\sqrt{2 \pi \sigma_w^2}} \right) ^{d+1} \cdot \exp\left( -\frac{1}{2 \sigma_w^2} \hat{W}^T \hat{W} \right) 
\end{aligned}
$$

选择一个 mode，即其最大的时候 $\hat{W}$ 取什么。所以忽略掉常数项，

$$
\begin{aligned}
&\max_{\hat{W}} \left(-\frac{1}{2\sigma^{2}}\sum_{i \in [n]} (y_i - \hat{W}^T \hat{x_i})^2 - \frac{1}{2 \sigma_w^2} || \hat{W} ||^{2}\right) \\
&\iff\\
&\min_{\hat{w}} \left( \sum_{i \in [n]}(y_i - \hat{W}^T \hat{x_i})^{2} + \lambda ||\hat{W}||^2 \right) 
\end{aligned}
$$

一开始对 $\hat{W}$ 的先验假设和 L2-Norm 起到的是同样的作用，L2-Norm 项可以看作是 $\hat{W}$ 偏移先验的惩罚。