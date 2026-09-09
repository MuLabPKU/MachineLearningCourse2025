---
title: 24秋机器学习笔记-06-学习理论
date: 2024-10-30 15:11:26
tags:
  - 本科课程
  - 机器学习
categories: [笔记, 本科课程, 机器学习]
---

本节内容假设模型为二分类，$y \in \{1,-1\}$，$x\in \mathcal{X}$。

## 问题引入

定义 $E_{\text{in}}$ 表示训练误差（in-sample error）。令 $h \in \mathcal{H}$ 为一个模型，例如 $h(x) = \operatorname{sgn}(w^T x + b)$。

$$
E_{\text{in}} = \frac{1}{n} \sum_{i \in [n]} 1(h(x_i)\neq y_i)
$$

此处 $1$ 为 indicator function，和艾弗森括号是一个意思。

定义 $E_{\text{out}}$ 为 out of sample error。**事实上，我们更关心 $E_{\text{out}}$ 而不是 $E_{\text{in}}$**。二者之间有某种 trade-off。

$$
E_{\text{out}}(h) = E_{(x,y) \sim P_{XY}} 1(h(x)\neq y) = P(h(x)\neq y)
$$

由这两个相减即得到 generalization error $= E_{\text{out}(h)} - E_{\text{in}}(h)$。

> 不同资料中对 generalization error 的定义不一样，有些认为是 $E_{\text{out}}(h)$，我们这里认为是 $E_{\text{out}}$ 与 $E_{\text{in}}$ 之差。所以不对这两个定义给出中文翻译了。

举例：对于两个学生

| Student | Exercise | Exam | Ein | Eout | GE |
| --- | --- | --- | --- | --- | --- |
| A | 100 | 40 | 0 | 0.6 | 0.6 |
| B | 80 | 80 | 0.2 | 0.2 | 0 |

> 如今深度学习模型的拟合能力都很强，所以一般不会出现 GE $<0$ 的情况。

我们的目标：使 GE 尽可能小。

事实上由于 $E_{\text{out}}$ 可以任意大（接近于 $1$），所以 GE 也是可以任意大的，所以我们无法对 GE 给出一个绝对的界。~~所以机器学习是伪学科（x）~~

## PAC 理论

我们没法给出一个绝对的界，但可以给出一个概率意义上的界。例如说：以一个相当高的概率（$1-\delta$）有 $E_{\text{out}}(h) - E_{\text{in}}(h) < \varepsilon(\mathcal{H},n,\delta)$。即这个所谓的 $\varepsilon$ 为模型空间、训练数据与 $\delta$ 的函数。这套语言称为 probably approximately correct (PAC) theory。

### Hoeffding 不等式

给定独立有界随机变量 $x_1, \cdots ,x_n$（不要求同分布），$x_i \in [a_i,b_i]$。

定义随机变量 $\bar{x}= \displaystyle \frac{1}{n} \sum_{i \in [n]}x_i$，则对 $\forall \varepsilon>0$，我们有

$$
P(\bar{x} - E[\bar{x}]\ge \varepsilon) \le \exp\left( - \frac{2n^2 \varepsilon^2}{\sum_{i \in [n]}(b_i - a_i)^{2}} \right) 
$$

且

$$
P(E[\bar{x}]-x\ge \varepsilon) \le \exp\left( - \frac{2n^2 \varepsilon^2}{\sum_{i \in [n]}(b_i - a_i)^{2}} \right) 
$$

证明超出本课范围，略过。

尝试理解：形式与中心极限定理很像。但不同的是，中心极限定理要求 $n\to +\infty$，而 Hoeffding 不等式是对 $\forall n$ 成立的。

$E[\bar{x}]-x$ 可以理解为 GE，$\varepsilon$ 在不等式两边作 trade off，$\varepsilon$ 越大，右边越小，反之亦然。$n$ 越大，右边越小（训练数据集大，自然更容易控制 GE），而 $b_i-a_i$ 越大，右边越大（$x_i$ 的值域越大，模型自然更难控制）

接下来，**对于一个固定的 $h$**，我们一定有

$$
P(\text{GE}\ge \varepsilon)\le \exp(-2n \varepsilon^{2})
$$

证明：根据 $E_{\text{in}}$ 的定义，$1(h(x_i)\neq y_i)$ 可以看成随机变量 $X_i$，$E_{\text{in}}$ 自然就是 $X_i$ 的均值 $\bar{X}$。而考虑 $E[\bar{X}]$，其即为

$$
\begin{aligned}
E[E_{\text{in}(h)}] &= \frac{1}{n} \sum_{i \in [n]} E_{(x_i,y_i)\sim P_{XY}} 1(h(x_i)\neq y_i) \\ &= E_{\text{out}}(h)
\end{aligned}
$$

而 $x_i$ 的上下界自然是 $[0,1]$，全部带入 Hoeffding 不等式就得到了上式。变换一下形式以贴近我们一开始想要的形式：以至少 $1- \exp(-2n \varepsilon ^{2})$ 的概率，有 $\text{GE}\le \varepsilon$。

再令 $\exp(-2n \varepsilon^{2}) = \delta$，则 $\varepsilon = \displaystyle \sqrt{\frac{1}{2n}\log \frac{1}{\delta}}$，更贴近了！但这个界并没有多大的意义，因为在推导的时候我们的前提是固定了模型 $h$，但是训练之前我们也并不知道会训练出一个什么样的 $h$。**not a practical bound**。训练造成的最严重的问题就是 $1(h(x_i) \neq  y_i)$ **不独立了**，于是 Hoeffding 不等式就用不了了。

> 疑问：对于固定的 $h$，相当于是 $\forall h$，但是训练得到的 $h$ 难道不是 $h$ 吗，为什么就不成立了？
> 
> 因为这个界是没见过训练数据就得到的 $h$，训练过程后 $h$ 是已经见过这个 sampled 训练数据的了，自然就不能套用上面的界了。

既然我们不能确定一个 $h$，那么就考虑给 $\mathcal{H}$ 一个界，这样对于 $\forall h$ 就都能成立了。

假设 $\mathcal{H}$ 为有限集 $\{h_1, \cdots ,h_m\}$，考虑 union bound $P(\exists h \in \mathcal{H}, \text{GE}(h)\ge \varepsilon)$。这里 $\exists h$ 一段的意思就是 $\text{GE}(h_1)\ge \varepsilon \text{ or } \text{GE}(h_2)\ge \varepsilon \text{ or }\cdots\text{ or } \text{GE}(h_m)\ge \varepsilon$。所以有

$$
P(\exists h \in \mathcal{H}, \text{GE}(h) \ge  \varepsilon) \le \sum_{i \in [m]} P(\text{GE}(h_i) \ge \varepsilon) \le m \exp(-2n \varepsilon^{2})
$$

于是现在有了第一个 practical PAC bound。直觉：$\mathcal{H}$ 越大，越难给出一个紧的界。尝试将其写成标准形式，令 $\delta = m \exp(-2n \varepsilon^{2})$，解得 $\varepsilon = \displaystyle \sqrt{\frac{1}{2n}\log \frac{m}{\delta}}$。即有至少 $1-\delta$ 的概率，$\displaystyle \forall h \in \mathcal{H}, \text{GE}(h) < \sqrt{\frac{1}{2n}\log \frac{m}{\delta}}$。

但是一般而言 $\mathcal{H}$ 不是有限集，例如某 RKHS 状物 $\mathcal{H} = \{h: h(x) = \operatorname{sgn}(w^Tx+b)\}$，此时无法使用 union bound。

但是可以发现，union bound 对每一个 $h$ 都统计一次 error，这个界明显过于松了。

> 待修。待修。待修。待修。待修。待修。待修。待修。待修。待修。待修。待修。待修。待修。待修。待修。

## 成长函数（growth function）

将某个 $h \in \mathcal{H}$ 应用在 $x_1,x_2, \cdots ,x_n \in \mathcal{X}$ 以得到 $n$ 元组 $(h(x_1),h(x_2), \cdots ,h(x_n))$，其被称为一个 dichotomy（二分）。

令 $\mathcal{H}(x_1,x_2, \cdots ,x_n):= \{(h(x_1), \cdots ,h(x_n)): h \in \mathcal{H}\}$，即对 $x_1, \cdots ,x_n$ 上作用所有 $h$ 能得到的所有 dichotomy 的集合。

接下来给出成长函数的定义：对于 $\mathcal{H}$，$m_{\mathcal{H}}(n)$ 被定义为

$$
m_{\mathcal{H}}(n) = \max_{x_1, \cdots ,x_n \in \mathcal{X}} \left| \mathcal{H}(x_1, \cdots ,x_n) \right| 
$$

其衡量的就是对于任意 $n$ 个 $\mathcal{X}$ 中的点，$\mathcal{H}$ 能生成的 dichotomy 的最大数量。

举例：对于线性模型和三个点，其可以生成 $2^3$ 种 dichotomy。

对任意 $\mathcal{H}$，一定有 $m_{\mathcal{H}}(n)\le 2^n$（上例就取到了上界）

若 $\mathcal{H}(x_1, \cdots ,x_n)$ 包含了 $\{x_1, \cdots ,x_n\}$ 的子集 $S$ 的所有可能的 dichotomy，则称 $\mathcal{H}(x_1, \cdots ,x_n)$ 打散了（shatters）$S$。例如 $\mathcal{H}(x_1,x_2,x_3) = \{(+1,-1,-1),(-1,+1,-1),(-1,+1,+1)\}$，则我们说 $\mathcal{H}$ shatter 了 $\varnothing,\{x_1\},\{x_2\},\{x_3\}$，但是更大的就不行了（例如 $\{x_1,x_2\}$）

举例：$\mathcal{X} = \mathbb{R}^{2}$，$\mathcal{H} = \{h:h(x) = \operatorname{sgn}(w^Tx+b)\}$，即所有的二维线性模型。显然 $m_{\mathcal{H}}(3)= 2^3 = 8$。对于 $m_{\mathcal{H}}(4)$ 呢？注意到有些 dichotomy 是一定无法生成的（画图便知），异或导致的第一次 AI 寒冬。$m_{\mathcal{H}}(4) = 14$。

再举例，$\mathcal{X}=\mathbb{R}$，$\mathcal{H} = \{h:h(x) = \operatorname{sgn}(x-a)\}$，则 $m_{\mathcal{H}}(n) = n+1<2^n~(n>1)$。

再考虑 $\mathcal{H} = \displaystyle \left\{ h: h(x) = \begin{cases} +1, & a\le x\le b \\ -1, &\text{otherwise}  \end{cases} \right\}$，显然 $\displaystyle m_{\mathcal{H}}(n) = \binom{n+1}{2}+1<2^n~(n>2)$。

## VC 维

使得 $m_{\mathcal{H}}(n) = 2^n$ 的最大的 $n$ 被称为 $\mathcal{H}$ 的 Vapnik-Chervonenkis dimension（VC 维），记作 $d_{\text{VC}}(\mathcal{H})$。

VC 维度量的为最大的 $\mathcal{H}$ 能 shatter 的样本量。小的 $d_{\text{VC}}$ 说明 $\mathcal{H}$ 的分类能力弱（考虑线性模型的 $d_{\text{VC}}$ 为 $3$），也意味着 hypothesis space $\mathcal{H}$ 不够大。相当于就是在度量 $\mathcal{H}$ 的有效维度。巧合地发现，二维线性模型恰好就有 $3$ 个参数。其实也就说明 VC 维与模型的参数是正相关的。

我们发现，可以利用 $d_{\text{VC}}(\mathcal{H})$ 给出 $m_{\mathcal{H}}(n)$ 的一个界。接下来给出 **Sauer's Lemma**：

$$
m_{\mathcal{H}}(n) \le \sum_{i=0}^{d_{\text{VC}}}\binom{n}{i}
$$

证明很复杂，略过。所以我们发现上界 $m_{\mathcal{H}}(n) = O(n^{d_{\text{VC}}})$，侧面说明这个引理的强大之处。

回顾：union bound，当 $|\mathcal{H}|=m$ 时，以至少 $1-\delta$ 的概率有 $\forall h \in \mathcal{H}, \text{GE}(h) < \displaystyle \sqrt{\frac{1}{2n} \log \frac{m}{\delta}}$。

若是将 $m$ 替换为 $m_{\mathcal{H}}(n)$，而 $m_{\mathcal{H}}(n)$ 若是只有 $2^n$ 的界，则取完 $\log$ 后分母上的 $n$ 也会不见，这不是我们想要的——我们希望 $n$ 越大，GE 的界能越小。

但是有了 Sauer's Lemma，$m_{\mathcal{H}}(n)$ 就有了 $O(n^{d_{\text{VC}}})$ 的上界，代入后也许就是我们想要的了。

## VC 泛化界

学习理论的重要结果。

我们不能直接将 $m$ 替换。

真实的结果为：有至少 $1-\delta$ 的概率，$\forall h\in \mathcal{H}$，$\text{GE}(h) < \displaystyle \sqrt{\frac{8}{n}\log \frac{4 m_{\mathcal{H}}(2n)}{\delta}}$（有一些常数的变化）。但是这个结果下，这个界就可以随着 $n$ 增大而降低了。

证明过于复杂，略。掌握直觉即可。

直觉就是：越大的 $n$，越小的 $d_{\text{VC}}$（模型越简单），GE 的界就会越紧！事实上就是各种 trade-off。