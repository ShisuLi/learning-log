
# 逻辑回归 (Logistic Regression) 终极笔记

## 1. 核心定位：名为回归，实为分类

逻辑回归是机器学习领域中最基础、最重要、应用最广泛的**分类算法**之一。尽管其名称中带有“回归”，但它解决的是**分类问题**，尤其是**二元分类 (Binary Classification)** 问题。

-   **目标**：预测一个离散的类别标签，例如：是/否、垃圾邮件/非垃圾邮件、患病/健康。
-   **输出**：模型输出的不是一个确定的类别，而是一个**概率值**，通常是样本属于“正类”（Positive Class, 通常记为 1）的概率。我们可以设定一个**阈值 (Threshold)**（例如 0.5），当概率大于阈值时，判定为正类；否则判定为负类。

---

## 2. 从线性回归到逻辑回归：为何不能用线性回归做分类？

一个自然的想法是：我们能否直接用线性回归 $y = w^Tx + b$ 的输出来做分类？比如，输出大于 0.5 就判为 1，小于 0.5 就判为 0。

**答案是：不可以。** 主要原因有二：

1.  **输出范围不匹配**：线性回归的输出是 $(-\infty, +\infty)$，而分类问题需要的概率输出应该在 $[0, 1]$ 区间内。线性模型的输出值远超这个范围，缺乏概率意义。
2.  **对离群值敏感**：如下图所示，当加入一个远离决策边界的离群点时，线性回归为了拟合这个新点，会使得拟合直线发生严重偏移，导致原有的决策边界被破坏，造成错误的分类。



为了解决这个问题，我们需要一个函数，能将线性回归的 unbounded 输出“压扁”到 $[0, 1]$ 区间内。这个函数就是 **Sigmoid 函数**。

---

## 3. Sigmoid 函数：连接线性和概率的桥梁

Sigmoid 函数，也称为 Logistic 函数，是逻辑回归的核心。

### 3.1 公式定义

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

其中 $z$ 是线性模型的输出，即 $z = w^Tx + b$。

### 3.2 函数图像与特性



1.  **值域**：函数的输出范围是 $(0, 1)$，完美地对应了概率的定义。
2.  **单调性**：函数是单调递增的。当 $z \to +\infty$ 时，$\sigma(z) \to 1$；当 $z \to -\infty$ 时，$\sigma(z) \to 0$。
3.  **中心对称**：函数关于点 $(0, 0.5)$ 中心对称。
4.  **优美的导数**：Sigmoid 函数的导数可以用其自身来表示，这在后续的梯度计算中非常方便。
    $$
    \frac{d}{dz}\sigma(z) = \sigma(z)(1 - \sigma(z))
    $$
    *推导*：
    $\sigma'(z) = \frac{d}{dz}(1+e^{-z})^{-1} = -1(1+e^{-z})^{-2} \cdot e^{-z} \cdot (-1) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1+e^{-z}-1}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} - \frac{1}{(1+e^{-z})^2} = \sigma(z) - \sigma(z)^2 = \sigma(z)(1-\sigma(z))$。

---

## 4. 模型表示与概率解释

将 Sigmoid 函数与线性模型结合，我们得到逻辑回归的**假设函数 (Hypothesis Function)** $h_\theta(x)$（这里用 $\theta$ 统一表示参数 $w$ 和 $b$）：

$$
h_\theta(x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}
$$

这个输出 $h_\theta(x)$ 有着明确的概率解释：它代表了给定输入 $x$ 和参数 $\theta$ 的条件下，样本类别 $y$ 为 1 的概率。

-   $P(y=1 | x; \theta) = h_\theta(x)$
-   $P(y=0 | x; \theta) = 1 - h_\theta(x)$

这两个式子可以优雅地合并成一个：

$$
P(y | x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}
$$

当 $y=1$ 时，上式为 $h_\theta(x)$；当 $y=0$ 时，上式为 $1 - h_\theta(x)$。

---

## 5. 损失函数：交叉熵 (Cross-Entropy)

如何衡量模型预测的好坏？我们需要一个损失函数。对于线性回归，我们用了均方误差 (MSE)。但 MSE 在逻辑回归中是一个**非凸函数 (non-convex)**，使用梯度下降优化时很容易陷入局部最优。

因此，我们从**最大似然估计 (Maximum Likelihood Estimation, MLE)** 的角度来推导逻辑回归的损失函数。

### 5.1 最大似然估计推导

我们的目标是找到一组参数 $\theta$，使得在我们给定的训练数据集（假设有 $m$ 个样本）上，观测到真实标签的**联合概率最大**。这个联合概率就是**似然函数 (Likelihood Function)** $L(\theta)$。

假设样本之间独立同分布，似然函数为所有样本概率的乘积：
$$
L(\theta) = P(\vec{y} | X; \theta) = \prod_{i=1}^{m} P(y^{(i)} | x^{(i)}; \theta) = \prod_{i=1}^{m} (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}}
$$
直接最大化这个乘积形式的函数很困难。一个标准的数学技巧是取对数，将连乘变为连加，同时不改变其单调性。这就得到了**对数似然函数 (Log-Likelihood)** $\ell(\theta)$：

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

在机器学习中，我们习惯于最小化**损失函数**，而不是最大化似然函数。损失函数通常是负的对数似然函数再求平均。因此，我们定义逻辑回归的损失函数 $J(\theta)$ 为**平均负对数似然**，这也就是**交叉熵损失函数**：

$$
J(\theta) = -\frac{1}{m} \ell(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

### 5.2 交叉熵的直观理解

-   当真实标签 $y=1$ 时，损失为 $-\log(h_\theta(x))$。如果预测概率 $h_\theta(x) \to 1$，损失 $\to 0$；如果预测 $h_\theta(x) \to 0$，损失 $\to +\infty$。
-   当真实标签 $y=0$ 时，损失为 $-\log(1 - h_\theta(x))$。如果预测概率 $h_\theta(x) \to 0$，损失 $\to 0$；如果预测 $h_\theta(x) \to 1$，损失 $\to +\infty$。

这种性质完美地惩罚了错误的预测，且损失函数 $J(\theta)$ 是一个**凸函数**，保证了梯度下降可以找到全局最优解。

---

## 6. 梯度下降求解

我们的目标是找到使 $J(\theta)$ 最小的参数 $\theta$。我们使用梯度下降来更新参数：
$$
\theta := \theta - \eta \nabla_\theta J(\theta)
$$
关键在于求解梯度 $\nabla_\theta J(\theta)$。我们先对单个样本的损失求梯度：
$$
\text{loss} = -[y \log(\sigma(z)) + (1-y)\log(1-\sigma(z))] \quad \text{其中 } z = w^Tx+b
$$
利用链式法则，我们求损失对权重 $w_j$（$x$ 的第 $j$ 个分量对应的权重）的偏导：
$$
\frac{\partial \text{loss}}{\partial w_j} = \frac{\partial \text{loss}}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial z} \cdot \frac{\partial z}{\partial w_j}
$$
1.  $\frac{\partial \text{loss}}{\partial \sigma} = -\left( \frac{y}{\sigma} - \frac{1-y}{1-\sigma} \right) = \frac{\sigma - y}{\sigma(1-\sigma)}$
2.  $\frac{\partial \sigma}{\partial z} = \sigma(1-\sigma)$
3.  $\frac{\partial z}{\partial w_j} = x_j$

将三者相乘：
$$
\frac{\partial \text{loss}}{\partial w_j} = \frac{\sigma - y}{\sigma(1-\sigma)} \cdot \sigma(1-\sigma) \cdot x_j = (\sigma(z) - y) x_j = (h_\theta(x) - y) x_j
$$
这个结果形式上惊人地简洁！它与线性回归的梯度形式 $(h_\theta(x)-y)x_j$ 完全一样，只是 $h_\theta(x)$ 的定义从 $w^Tx+b$ 变成了 $\sigma(w^Tx+b)$。

对所有 $m$ 个样本求平均，得到 $J(\theta)$ 对 $w_j$ 的梯度：
$$
\frac{\partial J(\theta)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$
梯度下降的更新规则即为：
$$
w_j := w_j - \eta \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

---

## 7. 决策边界 (Decision Boundary)

决策边界是用来将不同类别区域分开的超平面。在逻辑回归中，决策边界由 $w^Tx+b=0$ 定义。

-   当 $w^Tx+b > 0$ 时，$h_\theta(x) > 0.5$，预测为类别 1。
-   当 $w^Tx+b < 0$ 时，$h_\theta(x) < 0.5$，预测为类别 0。

**重要**：决策边界是**由参数 $w, b$ 决定的**，而不是由数据决定的。逻辑回归只能产生**线性决策边界**。要获得非线性决策边界，可以通过**特征映射**（多项式特征、核技巧等）将数据转换到更高维的空间，使其在高维空间中线性可分。

---

## 8. 正则化 (Regularization)

为了防止模型过拟合，我们可以在损失函数中加入正则化项。

-   **L2 正则化 (Ridge)**:
    $$
    J(\theta) = -\frac{1}{m} \sum \left[ \dots \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
    $$
    它倾向于让权重 $w$ 的值变得比较小，但不会变为 0。

-   **L1 正则化 (Lasso)**:
    $$
    J(\theta) = -\frac{1}{m} \sum \left[ \dots \right] + \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|
    $$
    它会产生**稀疏解**，即让许多不重要的特征权重变为 0，从而起到**特征选择**的作用。

---

## 9. 多分类逻辑回归 (Softmax Regression)

当类别数 $K > 2$ 时，我们可以使用 Softmax 回归，它是逻辑回归在多分类问题上的推广。

-   **Softmax 函数**：对于一个有 $K$ 个类别的分类问题，模型会对每个类别 $k$ 输出一个分数 $z_k$。Softmax 函数将这些分数转换为概率：
    $$
    P(y=k | x; \theta) = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
    $$
    其中 $z_k = w_k^T x + b_k$。

-   **损失函数**：同样使用**交叉熵损失**，但形式推广到多分类：
    $$
    J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} \mathbb{I}\{y^{(i)}=k\} \log \left( \frac{e^{z_k^{(i)}}}{\sum_{j=1}^{K} e^{z_j^{(i)}}} \right)
    $$
    其中 $\mathbb{I}\{\cdot\}$ 是指示函数，当条件成立时为 1，否则为 0。

---

## 10. 逻辑回归与其他模型的关系

-   **广义线性模型 (GLM)**：逻辑回归是广义线性模型的一种，其联系函数是 Logit 函数（$\log(\frac{p}{1-p})$）。
-   **神经网络**：一个不含隐藏层的、输出层使用 Sigmoid/Softmax 激活函数的神经网络，本质上就是逻辑回归/Softmax 回归。
-   **支持向量机 (SVM)**：
    -   **相似点**：两者都是线性分类器，都可用于二分类。
    -   **不同点**：SVM 的目标是找到使间隔最大化的决策边界，它只关心支持向量（离边界最近的点）；而逻辑回归考虑所有数据点，目标是最大化数据似然。SVM 输出的是类别，逻辑回归输出的是概率。

