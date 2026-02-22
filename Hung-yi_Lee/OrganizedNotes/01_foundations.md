# 1 基础与概率视角

## 1.1 机器学习三步走
- **定义函数族**：如线性回归/多层感知机，参数即候选函数。
- **度量好坏**：损失函数（回归常用 MSE，分类常用交叉熵）。
- **优化求最优**：梯度下降及其变体，得到在训练集上最优、并力求泛化的参数。

## 1.2 回归与分类的概率刻画
- **线性回归 = 高斯噪声假设下的 MLE**：误差 $\epsilon\sim\mathcal N(0,\sigma^2)$，最大化对数似然等价最小化 MSE。
- **逻辑回归 = 伯努利假设下的 MLE**：
  - 假设 $P(y=1|x)=\sigma(w^\top x+b)$，对数似然
    $$
    \ell(\theta)=\sum_{i}\big[y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\big]
    $$
    取负平均即**交叉熵损失**。
  - 梯度：$\frac{\partial J}{\partial w_j}=\frac{1}{m}\sum_i(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$。
- **Softmax 多分类**：$P(y=k|x)=\frac{e^{z_k}}{\sum_j e^{z_j}}$，损失为多分类交叉熵。
- **GLM 视角**：高斯→MSE，伯努利→交叉熵，泊松→对数似然 $L=\sum_i[h_\theta(x_i)-y_i\log h_\theta(x_i)]$。

## 1.3 梯度下降三兄弟
- **Batch GD**：每次用全部样本，方向准但慢。
- **SGD**：一次一个样本，噪声大但快，适合在线学习。
- **Mini-batch GD**：折衷，最常用。
更新式（批大小 $B$）：
$$
\theta\leftarrow\theta-\eta\frac{1}{B}\sum_{i\in\text{batch}}\nabla_\theta J^{(i)}
$$

## 1.4 优化器推导速览
- **Momentum**：$v^t=\lambda v^{t-1}-\eta\nabla_\theta L(\theta^{t-1})$，$\theta^{t}=\theta^{t-1}+v^t$，相当于历史梯度指数加权。
- **Adagrad**：为每个参数累积梯度平方 $G_t=\sum_{i=1}^t g_i\odot g_i$，更新
  $$
  \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t
  $$
  稀疏特征友好但学习率单调衰减。
- **RMSProp**：$S_t=\beta S_{t-1}+(1-\beta)(g_t\odot g_t)$，解决 Adagrad 衰减过快。
- **Adam**：一阶动量 $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$，二阶动量 $v_t=\beta_2 v_{t-1}+(1-\beta_2)(g_t\odot g_t)$，偏差校正 $\hat m_t,\hat v_t$ 后
  $$
  \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat v_t}+\epsilon}\hat m_t
  $$

## 1.5 线性模型的能力边界
- 单层感知机/逻辑回归只能给出**线性决策边界**，对 XOR 等非线性可分问题无能为力。
- 通过**特征映射**或**堆叠非线性（多层网络）**引入高阶特征，才能逼近任意连续函数（万能逼近）。

## 1.6 重要公式摘录
- Sigmoid 导数：$\sigma'(z)=\sigma(z)(1-\sigma(z))$。
- Tanh 导数：$1-\tanh^2(x)$。
- 交叉熵-Softmax 梯度：$\frac{\partial J}{\partial z_k}=p_k-\mathbb{I}[y=k]$。
