---
title: 24秋机器学习笔记-01-线性回归
date: 2024-09-11 23:09:12
tags:
  - 本科课程
  - 机器学习
categories: [笔记, 本科课程, 机器学习]
---

> 本文主要涉及线性回归（Linear Regression）
> [一键回城](/note-ml2024fall/)。


## 基本定义

记号约定：

- $D = \{(x_i,y_i)\}$ 为训练集，其中 $x_i \in \mathbb{R}^d,y\in \mathbb{R}$；
- 线性模型：$f(x) = w^Tx + b$，其中 $w\in \mathbb{R}^d,b \in R$，分别称为权重（weight）和偏置（bias）。
  $w$ 本质上是在对 $x$ 的每一维进行加权求和。

模型训练的原则：经验风险最小化（ERM, empirical risk minimization）

在做的事情就是最小化训练数据集上的损失。

定义损失函数（loss function）$L(w,b)$，在线性回归中我们使用平方损失（squared loss），表达式为：

$$
\min_{w,b} \frac{1}{n}\sum_{i \in [n]}(y_i - f(x_i))^{2}
$$

为了使这个表达式达到最小值，我们对其求梯度（gradient）：

$$
\begin{aligned}
\frac{\partial L(w,b)}{\partial w} &= -\sum_{i \in [n]}2(y_i - w^Tx_i - b)\cdot \frac{\partial (w^Tx_i)}{\partial w} \\
&= -\sum_{i \in [n]}2x_i(y_i - w^Tx_i - b) \\
\frac{\partial L(w,b)}{\partial b} &= -\sum_{i \in [n]}2(y_i - w^Tx_i - b)
\end{aligned}
$$

> 番外：矩阵/向量运算的求导
> 
> 常见的公式：
> - $\displaystyle \frac{\partial x^Tx}{\partial x} = 2x$
> - $\displaystyle \frac{\partial x^TAx}{\partial x} = (A + A^T)x$
> - $\displaystyle \frac{\partial a^Tx}{\partial x} = a$
>
> 原则是 $\displaystyle \frac{\partial f(x)}{\partial x}$ 的形状（shape）应与 $x$ 一致，根据这个原则可以凑（？）
> 更多可以查阅 *Matrix Cookbook*。

然后利用**梯度下降法**（gradient descent, GD）对 $w$ 和 $b$ 的值进行更新，具体地：

$$
\begin{aligned}
w' &\gets w - \alpha \cdot \frac{\partial L}{\partial w} \\
b' &\gets b - \alpha \cdot \frac{\partial L}{\partial b}
\end{aligned}
$$

其中 $\alpha$ 为**学习率**（learning rate, LR），是预先指定的**超参数**（hyperparameter），代表 $w,b$ 每次往梯度方向走的“步长”。

## 闭式解讨论

以上属于是数值方法，但对于线性回归而言，其是有**闭式解**（closed-form solution）的，我们就没有必要使用数值方法（其具有一定的随机性，且有不可避免的误差，此处不过多讨论）

为了方便，我们令 $\displaystyle X:=\begin{bmatrix} x_1^T & 1 \\ \vdots  & \vdots \\ x_n^T & 1 \end{bmatrix} \in \mathbb{R}^{n\times (d+1)}$，$\hat{W} := \begin{bmatrix} w \\ b \\\end{bmatrix} \in \mathbb{R}^{d+1}$，$y:= \begin{bmatrix} y_1 \\ \vdots \\ y_n \\\end{bmatrix} \in \mathbb{R}^n$，这样可以把参数都放进一个 $\hat{W}$ 里面，我们需要优化的目标就可以变成 $\left\| y - X\hat{W}^T \right\|^2$，为了对其求梯度，我们将其写成如下形式：

$$
\begin{aligned}
L(\hat{W}) &= (y - X \hat{W})^T(y - X \hat{W}) \\
&= y^Ty - y^TX \hat{W} - \hat{W}^T X^T y + \hat{W}^T X^T X \hat{W}\\
&= y^Ty - 2y^TX \hat{W} + \hat{W}^T X^T X \hat{W}\\
\frac{\partial L(\hat{W})}{\partial \hat{W}} &= -2 X^T y + 2 X^T X \hat{W}\\
&= -2 X^T(y - X \hat{W})
\end{aligned}
$$

令 $\displaystyle \frac{\partial L}{\partial \hat{W}} = 0$ 可以知道 $X^Ty = X^TX \hat{W}$。解的情况需要取决于 $X^TX$ 的可逆性。

- 若 $X^TX$ 可逆，则 $\hat{W} = (X^TX)^{-1} X^Ty$，这是最简单的情况。
- 若 $X^TX$ 不可逆，则 $X^TX$ 为奇异阵，一般有两种原因：
  - $d+1>n$，直觉上来看就是**数据点太少**，有不等式 $\operatorname{rank}(X^TX) = \operatorname{rank}(X) \le \min(n, d+1) = n < d + 1$。而 $X^TX \in \mathbb{R}^{(d+1)\times (d+1)}$，所以不可逆。这种情况一般比较罕见。
  - $d+1\le n$，直觉上来看是**有多余的特征维度**，$X$ 中有重复的列，导致不满秩。

而不论如何，若 $X^TX$ 不可逆，都意味着有多组 $\hat{W}$ 的可行解。下面证明解一定存在：

假设其无解，根据线性方程组的理论，说明 $\operatorname{rank}(X^TX) < \operatorname{rank}(X^TX \mid X^Ty)$，但这种情况显然不可能发生，于是线性回归的 $\hat{W}$ **要么解唯一，要么有无穷解**。*事实上，无穷的情况会比较不好处理*。

## 岭回归（Ridge Regression）

采用了 L2-正则化的线性回归称为岭回归。L2-正则化的意思是在损失函数后面追加一个正则化项 $\lambda \cdot \left\| \hat{W} \right\|^2$，以惩罚权重过大的模型（weight decay）

现在的损失函数为：

$$
L(\hat{W}) =\min_{\hat{W}}\left\{ (y - X \hat{W})^T (y - X \hat{W}) + \lambda \hat{W}^T \hat{W} \right\} 
$$

令其梯度为 $0$ 尝试推导闭式解：

$$
\begin{aligned}
-2X^T(y - X \hat{W}) + 2 \lambda \hat{W} &= 0 \\
X^Ty - X^TX \hat{W} &= \lambda \hat{W}\\
(X^TX + \lambda I) \hat{W} &= X^Ty
\end{aligned}
$$

接下来我们说明 $X^TX + \lambda I$ 一定是非奇异阵。

由于 $X^TX$ 为实对称矩阵，所以对其特征值分解可以得到

$$
X^TX = U \Lambda U^T = U \begin{bmatrix} \lambda_1 &  &  \\  & \ddots &  \\  &  & \lambda_{d+1} \\\end{bmatrix} U^T
$$

且 $X^TX$ 半正定（为什么，考虑 $\forall v \ne 0$，我们有 $v^T X^T X v = \left\| Xv \right\|^2\ge 0$），$\lambda_1 \ge \lambda 2 \ge \cdots \ge \lambda_{d+1} \ge  0$。

那我们对 $X^TX + \lambda I$ 也做同样操作：

$$
X^TX + \lambda I = U (\Lambda + \lambda I) U^T
$$

$\lambda > 0$，$\Lambda + \lambda I$ 是正定阵，于是 $X^TX + \lambda I$ 也是正定阵，一定可逆。

所以加上 L2-Norm 后，$\hat{W}$ 是一定有唯一解的：$(X^TX + \lambda I)^{-1} X^Ty$。

加上 L2-Norm 的好处还有一方面：考虑 $X^TX$ 的特征值分解 $U \operatorname{diag}\{\lambda_1, \cdots ,\lambda_{d+1}\}U^T$，一般而言有 $\lambda_{d+1}\to 0$。而 $(X^TX)^{-1} = U \Lambda^{-1} U^T = U \operatorname{diag} \{\lambda_1^{-1}, \cdots , \lambda_{d+1}^{-1}\}U^T$，数值稳定性就不太好。不过加上了正则化项之后，求完逆的最后一个特征值即为 $\displaystyle \frac{1}{\lambda_{d+1}+\lambda}$，不容易出现 numerical issues。

## Lasso 回归

L1-Norm：

$$
\min_{\hat{W}} L(\hat{W} ) + \lambda \left\| \hat{W} \right\|
$$

相当于希望 $\hat{W}$ 的大多数维度为空，即希望一个稀疏的 $\hat{W}$，可以理解为一种**特征选择**。

带上 L1 正则化项的线性回归称为 Lasso 回归（Least Absolute Shrinkage and Selection Operator）

## 最小二乘

另一种理解线性回归的方式。

理想情况下，我们希望 $X \hat{W} = y$。但事实是，$y$ 可能压根不在 $X$ 的列空间里面，所以并不存在这样的 $\hat{W}$。

那么我们自然希望找到一个 $\hat{y}$，满足 $\hat{y}$ 在 $\operatorname{col}(X)$ 里面，并且这个 $\hat{y}$ 与我们希望的 $y$“差距最小”。

所以，$y - \hat{y} \perp \operatorname{col}(X)$。即 $X^T(y - \hat{y}) = 0$，所以

$$
X^TX \hat{W} = X^Ty
$$

这与我们之前利用 ERM 推导的结果是相符的。