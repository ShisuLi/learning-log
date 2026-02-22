# 2 训练技巧与正则化

## 2.1 激活函数对比
- **Sigmoid**：$(0,1)$，零点对称，易梯度消失；导数 $\sigma'(x)=\sigma(x)(1-\sigma(x))$。
- **Tanh**：$(-1,1)$，零均值但仍有饱和区。
- **ReLU**：$f(x)=\max(0,x)$，正区间梯度恒1，缓解梯度消失；存在“死亡 ReLU”。
- **Leaky/Parametric ReLU**：负区间给小斜率 $\alpha$，PReLU 的 $\alpha$ 可学习。
- **Maxout**：$f(x)=\max(w_1^\top x+b_1,\ w_2^\top x+b_2,\dots)$，ReLU 为其特例，表达力强但参数多。

## 2.2 自适应学习率与动量
- **Adagrad/RMSProp/Adam**：见《01_foundations》，核心是用历史梯度二阶矩调节步长；Adam = RMSProp + Momentum + 偏差校正。
- **学习率规划**：$\eta$ 可随迭代衰减（如 $\eta/\sqrt{t}$），也可用余弦退火或 warmup。

## 2.3 正则化手段
- **L2（Weight Decay）**：$L_{\text{new}}=L+\frac{\lambda}{2}\|w\|_2^2$，更新 $\;w\leftarrow(1-\eta\lambda)w-\eta\nabla_w L$，抑制权重过大。
- **L1**：$L_{\text{new}}=L+\lambda\|w\|_1$，更新额外减去 $\eta\lambda\,\text{sgn}(w)$，促使稀疏。
- **Early Stopping**：监控验证集，性能恶化即停止，等效于隐式正则。

## 2.4 Dropout
- 训练时以概率 $p$ 随机置零神经元输出；每个 mini-batch 采样不同子网。
- 测试时使用全网，并将权重缩放为 $(1-p)$ 以匹配期望；可视作 $2^M$ 子网的近似集成。

## 2.5 训练/泛化问题
- **梯度消失/爆炸**：深层网络或 RNN 常见；可用 ReLU、残差结构、梯度裁剪、良好初始化缓解。
- **Batch Size**：小批有正则化效应，过大易欠拟合；需结合学习率调节。
- **过拟合**：模型深度/参数过多或数据不足时出现；用正则化、数据增强、减小深度或增加数据缓解。
