---
title: 24秋机器学习笔记-05-表示定理
date: 2024-10-23 15:11:45
tags:
  - 本科课程
  - 机器学习
categories: [笔记, 本科课程, 机器学习]
---

## 引例

课上通过如下几个例子引出表示定理相关内容。

### SVM

在带松弛变量的 SVM 表示中，

$$
\min_w \sum_{i \in [n]} \max(0, 1-y_i(w^T x_i+b)) + \lambda \left\| w \right\|^{2}
$$

接下来将记号进行改写（本节内容均如此约定）：换成 $f(x_i) = w^T \varphi(x_i)$，优化形式变为

$$
\min_w \sum_{i \in [n]} \max(0, 1-y_i(w^T \varphi(x_i))) + \lambda \left\| w \right\|^{2}
$$

（注意这里的 $w$ 包含之前的 $b$，多进行了对 $b$ 的 norm，但实际上不太影响）

为了推导对偶形式，将其再变回带约束的优化问题，此时重新引回 $\xi_i = \max(0, 1-y_i w^T \varphi(x_i))$，问题变为

$$
\min_{w,\xi} \sum_{i\in [n]}\xi_i + \frac{\lambda}{2} \left\| w \right\|^{2}\\
\text{s. t. } \begin{cases} \xi_i\ge 0  \\ \xi_i\ge 1 - y_iw^T\varphi(x_i) \end{cases} ,\forall i \in [n]
$$

此处 $\lambda$ 除以二是为了方便求导。

引入两套拉格朗日乘子，写出拉格朗日函数：

$$
L(w,\xi,\alpha,\beta) = \sum_{i \in [n]}\xi_i+\frac{\lambda}{2} w^Tw + \sum_{i \in [n]}\alpha_i(1 - y_iw^T \varphi(x_i) - \xi_i) - \sum_{i \in [n]}\beta_i \xi_i
$$

对偶问题则为：

$$
\max_{\alpha\ge 0,\beta\ge 0} \min_{w,\xi} L(w,\xi,\alpha,\beta)
$$

根据 K.K.T. 条件，里层要达到最小，则偏导为 $0$：

$$
\begin{aligned}
\frac{\partial L}{\partial w} &= \lambda w - \sum_{i \in [n]}\alpha_i y_i \varphi(x_i) = 0 \\
\frac{\partial L}{\partial \xi_i} &= 1 - \alpha_i - \beta_i = 0
\end{aligned}
$$

这推出：

$$
\begin{aligned}
w &= \frac{1}{\lambda} \sum_{i \in [n]}\alpha_i y_i \varphi(x_i) \\
\alpha_i + \beta_i &= 1, & \forall i\in [n]
\end{aligned}
$$

这说明 $w$ 可以写成训练集数据经过某种训练后的线性组合。

带入外层并化简，发现 $\xi_i$ 的项都没了，注意把 $w$ 换掉：

$$
\max_{\alpha,\beta}\left\{ -\frac{1}{2\lambda}\left( \sum_{i \in [n]}\alpha_i y_i \varphi(x_i)^T \right) \left( \sum_{j \in [n]} \alpha_j y_j \varphi(x_j) \right) + \sum_{i \in [n]}\alpha_i\right\}\\
\text{s. t. }\alpha_i\ge 0,\beta_i\ge 0,\alpha_i+\beta_i=1
$$

$\beta_i$ 多余，

$$
\max_{\alpha} \sum_{i \in [n]}\alpha_i - \frac{1}{2\lambda}\sum_{i \in [n]}\sum_{j \in [n]}\alpha_i \alpha_j y_i y_j \varphi(x_i)^T \varphi(x_j)\\
\text{s. t. }0\le \alpha_i\le 1, \forall i
$$

与原先的区别：$\lambda$ 与 $\alpha_i\le 1$ 的上界，且没有 $\sum \alpha_i y_i = 0$。从直觉上来说，$\alpha_i$ 表示支持的“强度”，hard constraint 的时候，outlier 的 $\alpha_i$ 就会很大，引入松弛变量后，就不会有某一个单独的向量（outlier）贡献特别大的 $\alpha_i$。

### 岭回归

目标函数：

$$
\min_w \frac{1}{2}\sum_{i \in [n]}(y_i - w^T \varphi(x_i))^{2}+\frac{\lambda}{2} \left\| w \right\|^{2}
$$

对 $w$ 求偏导：

$$
\frac{\partial L}{\partial w} = -\sum_{i \in [n]}(y_i - w^T \varphi(x_i))\varphi(x_i)+\lambda w = 0
$$

所以

$$
w = \frac{1}{\lambda}\sum_{i \in [n]}(y_i - w^T \varphi(x_i))\varphi(x_i)
$$

定义 $\alpha_i = \displaystyle \frac{1}{\lambda}(y_i - w^T \varphi(x_i))$，则 $w = \displaystyle \sum_{i \in [n]} \alpha_i \varphi(x_i)$，注意这里只是说明 $w$ 满足此形式，即可以被表示为 $\varphi(x_i)$ 的线性组合。

### 逻辑回归 + L2-Norm

假设 $y \in \{-1,1\}$

$$
\min_w \sum_{i \in [n]} \log(1 + \exp(-y_i w^T \varphi(x_i))) + \frac{\lambda}{2}\left\| w \right\|^{2} 
$$

一样，求偏导：

$$
\frac{\partial L}{\partial w} = -\sum_{i \in [n]} \frac{\exp(-z_i)}{1+\exp(-z_i)}y_i \varphi(x_i)+\lambda w = 0
$$

所以

$$
w = \frac{1}{\lambda}\sum_{i \in [n]}\sigma(-z_i) y_i \varphi(x_i)
$$

令 $\displaystyle \alpha_i = \frac{1}{\lambda} \sigma(-z_i)y_i$，则 $\displaystyle w = \sum_{i \in [n]}\alpha_i \varphi(x_i)$。

如果要预测一个点的话，$f(x) = \displaystyle w^T \varphi(x) = \sum_{i \in [n]}\alpha_i \varphi(x_i)^T\varphi(x) = \sum_{i \in [n]} \alpha_i k(x_i,x)$，这不是偶然的。

### 一般性推测

根据上面的例子，我们可以猜想，对于任意线性模型 + ERM + 正则化都可以有

$$
f(x) = \sum_{i \in [n]}\alpha_i k(x_i,x)
$$

此为**表示定理**。为了进一步说明，引入一些数学工具。

## 再生核希尔伯特空间（RKHS）

在课上听这个的时候有点懵，这里进行重新整理。

**由于本节内容不做考察要求，且作者的数学并不太好，所以下面的定义与推导都不严谨，如有错误敬请指出**

### Hilbert 空间

一个希尔伯特空间（Hilbert Space）是对欧氏空间的一个扩展，其还是向量空间，有内积 $\langle f,g\rangle_{\mathcal{H}}$。由内积可以定义范数 $\left\| f \right\| := \sqrt{\langle f,f\rangle_{\mathcal{H}}}$，且*完备*（no holes，每个柯西列都收敛到某个点）。

内积要满足的性质：

- 对称性：$\langle f,g\rangle_{\mathcal{H}}=\langle g,f\rangle_{\mathcal{H}}$；
- 线性性：$\langle af_1+bf_2,g\rangle_{\mathcal{H}} = a\langle f_1,g\rangle_{\mathcal{H}} + b\langle f_2,g\rangle_{\mathcal{H}}$；
- 正定性：$\langle f,f\rangle_{\mathcal{H}}\ge 0$ 且 $\langle f,f\rangle_{\mathcal{H}}=0 \iff f = 0$

事实上，定义在集合 $\mathcal{X}$ 上的全体函数 $f: \mathcal{X}\to \mathbb{R}$ 便可以构成一个希尔伯特空间（函数可以看作为无穷维度的向量，回忆高代书上的内容（x））

### RKHS

> 定义（from Wikipedia）：考虑一个希尔伯特空间，其元素为定义在 $\mathcal{X}$ 上的函数，若 $\forall x \in \mathcal{X}$，存在函数 $K_x \in \mathcal{H}$ 使得 $\forall f \in \mathcal{H}$ 有 $f(x) = \langle f, K_x \rangle_{\mathcal{H}}$，则 $\mathcal{H}$ 为再生核希尔伯特空间（RKHS）

有点抽象，我们换个角度重新理解，因为理论上这个所谓的 RKHS 就是由核“再生”出来的，所以我们自然从核函数考虑起会自然一些。

考虑一个合法的核函数 $K(x,y)$，由于我们可以将函数看成无穷维向量，此二元函数自然就可以看成无穷维矩阵。其正定性可以被解读为 $\displaystyle \int\int f(x)K(x,y)f(y) \mathrm{d}x\mathrm{d}y\ge 0$，对称性可解读为 $K(x,y) = K(y,x)$。

那么同样地，类似对普通矩阵特征值分解，我们也可以对核函数进行类似操作，即存在特征值 $\lambda$ 与特征函数 $\psi(x)$ 使得

$$
\int K(x,y) \psi(x) \mathrm{d}x = \lambda \psi(x)
$$

特征向量自然是正交的（证明略过），即

$$
\langle \psi_1,\psi_2 \rangle_{\mathcal{H}} = \int \psi_1(x)\psi_2(x)\mathrm{d}x = 0
$$

这说明一个核函数对应着无穷个特征值与无穷个特征方程，与矩阵类似地对其特征值分解：

$$
K(x,y) = \sum_{i=1}^{\infty} \lambda_i \psi_i(x) \psi_i(y)
$$

这就是 SVM 一文中所说的 Mercer 定理，这说明所有的 $\psi_i$ 构成了函数空间的一组正交基。

$\{\sqrt{\lambda_i}\psi_i\}_{i=1}^{\infty}$ 自然为空间中的一组正交基。用这组基可以张成希尔伯特空间 $\mathcal{H}$，该空间的任意一个函数都可以表示为这组基的线性组合：

$$
f = \sum_{i=1}^{\infty} f_i \sqrt{\lambda_i} \psi_i
$$

自然，$f$ 就可以被这组基表示为 $(f_1,f_2, \cdots )$。若另一个函数 $g$ 可以被表示为 $(g_1,g_2, \cdots )$，则 $f$ 与 $g$ 的内积就可以表示为

$$
\langle f,g \rangle_{\mathcal{H}} = \sum_{i=1}^{\infty} f_i g_i
$$

对于核函数 $K$，固定一个变量后的 $K(x,\cdot )$ 为一元函数，用“特征值分解”的等式拆开：

$$
K(x,\cdot ) = \sum_{i=1}^{\infty} \sqrt{\lambda_i} \psi_i(x) \sqrt{\lambda_i} \psi_i = \sum_{i=1}^{\infty} \lambda_i \psi_i(x) \psi_i
$$

用上述基可以表示为 $(\sqrt{\lambda_1}\psi_1(x),\sqrt{\lambda_2}\psi_2(x), \cdots )$。

同理，$K(\cdot ,y)$ 可以表示为 $(\sqrt{\lambda_1}\psi_1(y), \sqrt{\lambda_2}\psi_2(y), \cdots )$，所以，这二者的内积

$$
\langle K(x,\cdot ), K(\cdot ,y) \rangle_{\mathcal{H}} = \sum_{i=1}^{\infty} \lambda_i \psi_i(x) \psi_i(y) = K(x,y)
$$

这就是核的**可再生性**，$\mathcal{H}$ 即为 RKHS。

那么对于之前的 $f = \displaystyle \sum_{i=1}^{\infty} f_i \sqrt{\lambda_i} \psi_i$，显然就有 $\displaystyle f(x) = \sum_{i=1}^{\infty} f_i \sqrt{\lambda_i} \psi_i(x)$，

$$
\langle K(x,\cdot ),f \rangle_{\mathcal{H}} = \sum_{i=1}^{\infty} \sqrt{\lambda_i}\psi_i(x) \cdot f_i = f(x)
$$

感性理解一下，就是这个 kernel 决定了这个希尔伯特空间。这使得函数的求值可以被等价为在原空间 $\mathcal{X}$ 中的某种 similarity。$k$ 从 $\mathcal{X}$ “reproduces” 每个 $f \in \mathcal{H}$。

很多函数空间都是 RKHS：

1. 考虑 $f(x) = w^Tx$（先忽略偏置项），所有的 $f$ 就组成了一个希尔伯特空间 $\mathcal{H}$。考虑 $k(z,x) = z^Tx$，$f(x) = \langle k(w, \cdot ), k(\cdot , x)\rangle_{\mathcal{H}} = k(w,x) = w^Tx$。这个核函数将 $w$ 映射到 $f$。考虑 $\left\| f \right\|^{2}$，其为 $\langle k(w,\cdot ),k(\cdot ,w)\rangle_{\mathcal{H}} = k(w,w) = \left\| w \right\|^{2}$。

2. 考虑 $f(x) = w^T \varphi(x)$（更为普遍的形式），定义核函数 $k(z,x) = \varphi(z)^T \varphi(x)$。因为 $f(x) = \langle k(\varphi^{-1}(w),\cdot ),k(\cdot ,x)\rangle_{\mathcal{H}} = k(\varphi^{-1}(w),x) = w^T \varphi(x)$。考虑 $\left\| f \right\|^{2}$，$\langle k(\varphi^{-1}(w),\cdot ), k(\cdot , \varphi^{-1}(w))\rangle_{\mathcal{H}} = w^Tw$。

> 所以其实对 $w$ 正则化和对 $f$ 正则化是等价的


## 表示定理

### 定理内容

考虑 RKHS $\mathcal{H}$ 与其再生核 $k:\mathcal{X}\times \mathcal{X}\to \mathbb{R}$，给定训练数据 $\{(x_1,y_1), \cdots ,(x_n,y_n)\} \subseteq \mathcal{X} \times \mathbb{R}$，损失函数 $L: \mathbb{R}\times \mathbb{R}\to \mathbb{R}$，正则化项 $R:[0,+\infty)\to \mathbb{R}$，且其严格递增。

对于任意使得 $\displaystyle \sum_{i \in [n]}L(f(x_i),y_i) + R(\left\| f \right\|)$ 最小的 $f \in \mathcal{H}$，都可以被表示成

$$
f = \sum_{i \in [n]} \alpha_i k(x_i, \cdot )
$$

### 证明

先将 $f$ 分解为两部分：一部分在 $\langle k(x_1,\cdot ),k(x_2,\cdot ), \cdots k(x_n,\cdot ) \rangle $ 上（此处记号为 span），另一部分 $v$ 与前一部分正交。于是 $\forall f$ 可以写成 $\displaystyle  f = \sum_{i \in [n]} \alpha_i k(x_i,\cdot ) + v$，满足 $\langle v,k(x_i,\cdot ) \rangle_{\mathcal{H}} = 0 $。接下来证明 $v$ 一定为 $0$ 即可。

考虑一个训练样本点 $x_j$，有 

$$
\begin{aligned}
f(x_j) &= \left\langle \sum_{i \in [n]}\alpha_ik(x_i,\cdot ) + v, k(\cdot ,x_j) \right\rangle_{\mathcal{H}}\\
&= \sum_{i \in [n]}\alpha_i \langle k(x_i,\cdot ), k(\cdot ,x_j) \rangle_{\mathcal{H}} + \sum_{i \in [n]}\langle v, k(\cdot ,x_j) \rangle_{\mathcal{H}}\\
&= \sum_{i \in [n]} \alpha_i k(x_i, x_j)
\end{aligned}
$$

这进而说明损失函数是跟 $v$ 无关的。

考虑 $R(\left\| f \right\|) = R(\left\| f_0+v \right\|)$。$\left\| f_0+v \right\|^{2} = \langle f_0,f_0 \rangle_{\mathcal{H}} + \langle v,v \rangle_{\mathcal{H}} + 2\langle f_0,v\rangle_{\mathcal{H}} \ge \left\| f_0 \right\|^{2}$。而且等于号成立当且仅当 $v=0$。又 $R$ 严格递增，所以为了使得整体的值最小（损失函数项与 $v$ 无关），我们自然是需要 $v=0$ 的。$\square$

直觉上理解起来应该挺自然的，模型的输出结果必然是与训练数据有关的，而正则化项把无关项去掉了。

### 优势

- 将一个对 $f$ 的潜在的无穷维优化变成只对 $n$ 个变量的搜索。
  考虑 RBF 核，$\varphi(x)$ 是无穷维的，$w^T$ 也应无穷维，这样就无法优化了，但利用表示定理我们就可以对 $n$ 个 $\alpha_i$ 进行优化了。
- 表明相当一部分机器学习算法的解能被有限训练数据上的核函数的线性组合表出（搭建参数化与非参数化的桥梁）。

## 线性回归的对偶形式

下节课的东西。