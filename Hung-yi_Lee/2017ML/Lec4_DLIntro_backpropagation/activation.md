## Improve Results on Training Data
### New Activation Functions
- Sigmoid: 梯度消失，靠近input的权重改变时对output影响不大
    $$\sigma(x) = \frac{1}{1+e^{-x}}\\ \sigma'(x) = \sigma(x)(1-\sigma(x))$$
- Tanh: 梯度消失，靠近input的权重改变时对output影响不大
    $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}\\ \tanh'(x) = 1 - \tanh^2(x)$$
- ReLU：正值梯度为1，负值梯度为0
    $$f(x) = \max(0,x)\\ f'(x) = \begin{cases}1 & x>0\\0 & x\leq0\end{cases}$$
- Leaky ReLU
    $$f(x) = \begin{cases}x & x>0\\\alpha x & x\leq0\end{cases}\\ f'(x) = \begin{cases}1 & x>0\\\alpha & x\leq0\end{cases}$$
- Parametric ReLU
    $$f(x) = \begin{cases}x & x>0\\\alpha x & x\leq0\end{cases}\\ f'(x) = \begin{cases}1 & x>0\\\alpha & x\leq0\end{cases}$$
    - $\alpha$ is learnable parameter
- Maxout: learnable activation function
    $$f(x) = \max(w_1 x + b_1, w_2 x + b_2)\\ f'(x) = \begin{cases}w_1 & w_1 x + b_1 > w_2 x + b_2\\w_2 & w_1 x + b_1 \leq w_2 x + b_2\end{cases}$$
    - ReLU is a special case of Maxout with 2 linear functions, one of which is always 0
    $$f(x) = \max(x, 0) = \max(1 \cdot x + 0, 0 \cdot x + 0)$$

### Adaptive Learning Rate
- Adagrad: Use first derivative to estimate second derivative
    $$w^{t+1} \leftarrow w^t - \frac{\eta}{\sqrt{\sum^t_{i=0}(g^i)^2}} g_t$$
- RMSProp: Use moving average to estimate second derivative
    $$w^1 \leftarrow w^0 - \frac{\eta}{\sigma^0}g^0,\quad \sigma^0=g^0\\
    w^2 \leftarrow w^1 - \frac{\eta}{\sigma^1}g^1,\quad \sigma^1=\sqrt{\alpha (\sigma^0)^2 + (1-\alpha)(g^1)^2}\\ w^3 \leftarrow w^2 - \frac{\eta}{\sigma^2}g^2,\quad \sigma^2=\sqrt{\alpha (\sigma^1)^2 + (1-\alpha)(g^2)^2}\\ \cdots\\w^{t+1} \leftarrow w^t - \frac{\eta}{\sigma^t}g^t,\quad \sigma^t=\sqrt{\alpha (\sigma^{t-1})^2 + (1-\alpha)(g^t)^2}$$
- Momentum: movement of last step minus gradient at present
    - Start at point  $\theta^0$
    - Movement $v^0=0$
    - Compute gradient at $\theta^0$
    - Movement $v^1 = \lambda v^0 - \eta \nabla L(\theta^0)$
    - Move to $\theta^1 = \theta^0 + v^1$
    - Compute gradient at $\theta^1$
    - Movement $v^2 = \lambda v^1 - \eta \nabla L(\theta^1)$
    - Move to $\theta^2 = \theta^1 + v^2$
    - Repeat until convergence
    - $v^i$ is actually the weighted sum of all the previous gradient: $$v^0 = 0\\ v^1 = -\eta \nabla L(\theta^0)\\ v^2 = -\eta (\lambda \nabla L(\theta^0) + \nabla L(\theta^1))\\ v^3 = -\eta (\lambda^2 \nabla L(\theta^0) + \lambda \nabla L(\theta^1) + \nabla L(\theta^2))\\ \cdots\\ v^t = -\eta \sum^{t-1}_{i=0} \lambda^{t-1-i} \nabla L(\theta^i)$$
    - Still not guarantee reaching global minimum, but give some hope.

- Adam: RMSProp + Momentum

## Improve Results on Test Data

### Early Stopping

- Split training data into training set and validation set
- Train on training set, evaluate on validation set
- Stop training when performance on validation set starts to degrade

### Regularization

- Find a set of weight not only minimizing original cost but also close to zero
- L2 Regularization (Weight Decay, usually not consider biases):
    $$L_{new}(\theta) =L_{original}(\theta) + \frac{1}{2}\lambda ||\theta||_2 =L_{original}(w) + \frac{1}{2}\lambda \sum_i w^2\\ \text{Gradient:} \frac{\partial L_{new}}{\partial w} = \frac{\partial L_{original}}{\partial w} + \lambda w\\\text{Update:} w^{t+1} \leftarrow w^t - \eta \left(\frac{\partial L_{original}}{\partial w} + \lambda w\right)= \left(1 - \eta \lambda\right) w^t - \eta \frac{\partial L_{original}}{\partial w}$$
    - Weight decay term $\left(1 - \eta \lambda\right) w^t$ shrink weights towards zero

- L1 Regularization:
    $$L_{new}(\theta) =L_{original}(\theta) + \lambda ||\theta||_1 =L_{original}(w) + \lambda \sum_i |w|\\ \text{Gradient:} \frac{\partial L_{new}}{\partial w} = \frac{\partial L_{original}}{\partial w} + \lambda \text{sgn}(w)\\\text{Update:} w^{t+1} \leftarrow w^t - \eta \left(\frac{\partial L_{original}}{\partial w} + \lambda \text{sgn}(w)\right)= w^t - \eta \frac{\partial L_{original}}{\partial w} - \eta \lambda \text{sgn}(w)$$
    - L1 regularization tends to produce sparse weights

### Dropout
- Each time before updating the parameters, each neuron has p% chance to be temporarily dropped out (set output to zero). Using the new network to do forward and backward propagation.
- For each mini-batch, we resample the dropout neurons.

- At test time, use the full network but scale down the weights by p%(all the weights times (1-p)%) to account for the fact that more neurons are active.
- Assume that the dropout rate is 50%. If a weight w=1 by training, set w=0.5 for testing.

- Dropout is a kind of ensemble
    - Train a bunch of networks with different structures
    - At test time, average their predictions (approximately done by scaling down weights)

- Training of Droptout: M neurons--2^M possible networks
    - Using one mini-batch to train one of the 2^M networks
    - Some parameters are shared among different networks

- Testing of Dropout:
    - Average the predictions of all 2^M networks, but not feasible
    - Approximate by using the full network with weights scaled down by p% (All the weights times (1-p)%)
    - Neural Network is not linear. Not equivalent but works well in practice 

好的，这是一份对您提供的深度学习笔记的详细翻译、扩展和讲解。笔记的核心是围绕两个关键问题展开的：**如何提升模型在训练集上的表现（即如何让模型学得更好）**，以及**如何提升模型在测试集上的表现（即如何让模型泛化得更好）**。

我将按照您笔记的结构，对每个知识点进行深入的剖析和补充，并以更具可读性的方式组织。

---

## Part 1: 提升训练集表现 (Improve Results on Training Data)

这部分的目标是解决**欠拟合 (Underfitting)** 的问题，即模型因为不够强大或训练方法不当，无法充分学习训练数据中的模式。

### 1.1 激活函数 (Activation Functions)

激活函数是神经网络的灵魂，它引入**非线性**，使得网络能够学习和拟合复杂的非线性关系。如果不用激活函数，多层神经网络本质上就等同于一个单层的线性模型。

#### 1.1.1 Sigmoid

*   **公式**:
    $$
    \sigma(x) = \frac{1}{1+e^{-x}}
    $$
*   **导数**:
    $$
    \sigma'(x) = \sigma(x)(1-\sigma(x))
    $$
*   **讲解**:
    *   **优点**:
        1.  **输出平滑**: 输出值在 (0, 1) 之间，可以被解释为概率或神经元的“激活率”。
        2.  **可微性**: 在所有点上都有导数。
    *   **缺点 (致命的)**:
        1.  **梯度消失 (Vanishing Gradient)**:
            *   从导数图像可以看出，当输入 `x` 的绝对值很大时（例如大于 5 或小于 -5），导数 $\sigma'(x)$ 会趋近于 0。
            *   在反向传播中，根据链式法则，梯度会逐层相乘。如果网络很深，多个接近 0 的梯度相乘，会导致靠近输入层的梯度变得极小，几乎为零。
            *   **后果**: 靠近输入层的权重几乎无法更新，模型训练停滞。这就是笔记中提到的“靠近 input 的权重改变时对 output 影响不大”。
        2.  **输出非零中心 (Not Zero-Centered)**: Sigmoid 的输出恒为正。这会导致后续层的输入是非零中心的，在反向传播时可能导致梯度更新方向的效率降低（产生所谓的 "zig-zagging" 路径），收敛变慢。

#### 1.1.2 Tanh (双曲正切函数)

*   **公式**:
    $$
    \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1
    $$
*   **导数**:
    $$
    \tanh'(x) = 1 - \tanh^2(x)
    $$
*   **讲解**:
    *   **优点**:
        1.  **输出是零中心的**: Tanh 的输出范围是 (-1, 1)，解决了 Sigmoid 的非零中心问题，通常能带来更快的收敛速度。
    *   **缺点**:
        1.  **梯度消失问题依然存在**: 当输入 `x` 的绝对值很大时，其导数同样趋近于 0，因此深层网络中仍然会面临梯度消失的问题。

#### 1.1.3 ReLU (Rectified Linear Unit, 修正线性单元) - **现代神经网络的默认选择**

*   **公式**:
    $$
    f(x) = \max(0, x)
    $$
*   **讲解**:
    *   **优点**:
        1.  **解决了梯度消失问题 (在正区间)**: 当输入 $x > 0$ 时，梯度恒为 1。这意味着只要神经元被激活，梯度就可以无衰减地在网络中传播。
        2.  **计算极其高效**: 只涉及一个简单的比较操作，远快于 Sigmoid/Tanh 的指数运算。
        3.  **稀疏性 (Sparsity)**: 当输入 $x \le 0$ 时，神经元输出为 0。这会使网络中的一部分神经元“死亡”，形成稀疏的激活状态，这在一定程度上更符合生物学，并且可能有助于降低过拟合。
    *   **缺点**:
        1.  **Dying ReLU 问题**: 如果一个神经元的输入恒为负，那么它的输出将永远是 0，梯度也永远是 0。这个神经元就“死亡”了，其权重再也无法更新。不恰当的学习率设置很容易导致大量神经元死亡。
        2.  **非零中心**: 输出同样是非零中心的。

#### 1.1.4 Leaky ReLU (带泄露的 ReLU)

*   **公式**:
    $$
    f(x) = \begin{cases} x & x>0 \\ \alpha x & x \le 0 \end{cases} \quad (\text{其中 } \alpha \text{ 是一个很小的常数, 如 0.01})
    $$
*   **讲解**:
    *   **目的**: 为了解决 Dying ReLU 问题。当输入为负时，不再输出 0，而是给它一个很小的负斜率 $\alpha$。这样即使神经元输入为负，它依然有梯度，依然可以学习。

#### 1.1.5 Parametric ReLU (PReLU, 参数化 ReLU)

*   **公式**:
    $$
    f(x) = \begin{cases} x & x>0 \\ \alpha x & x \le 0 \end{cases} \quad (\text{其中 } \alpha \text{ 是一个可学习的参数})
    $$
*   **讲解**:
    *   PReLU 是 Leaky ReLU 的一个变体，它更进一步，将负区间的斜率 $\alpha$ 作为一个**可学习的参数**，让网络自己去决定最佳的斜率应该是多少。

#### 1.1.6 Maxout

*   **公式**:
    $$
    f(x) = \max(w_1^T x + b_1, w_2^T x + b_2, \dots, w_k^T x + b_k)
    $$
*   **讲解**:
    *   Maxout 是一种**可学习的激活函数**。它本身就是一个小型的网络层。它将多个线性变换的结果取最大值作为输出。
    *   **优点**: 非常强大和灵活，它可以拟合任何凸函数。它也继承了 ReLU 的优点（无饱和、梯度不消失），同时避免了 Dying ReLU 问题。
    *   **缺点**: 每个 Maxout 单元的参数数量是普通神经元的 $k$ 倍，大大增加了模型的总参数量。
    *   **与 ReLU 的关系**: ReLU 是 Maxout 的一个特例。如笔记中所述，当 $k=2$, $w_1=I, b_1=0, w_2=0, b_2=0$ 时，Maxout 就等价于 ReLU。

### 1.2 自适应学习率 (Adaptive Learning Rate)

优化器的核心目标是指导模型参数（权重）如何更新，以最快最稳的方式找到损失函数的最小值。

#### 1.2.1 Adagrad

*   **思想**: 为每个参数独立地适应学习率。它根据参数的历史梯度大小来调整学习率：**历史梯度越大的参数，其学习率衰减得越快**。
*   **公式解读**:
    $$
    w^{t+1} \leftarrow w^t - \frac{\eta}{\sqrt{\sum^t_{i=0}(g^i)^2 + \epsilon}} g_t
    $$
    *   $\sum^t_{i=0}(g^i)^2$: 从开始到现在的梯度平方的累加和。
    *   分母 $\sqrt{\sum^t_{i=0}(g^i)^2 + \epsilon}$ 会随着训练的进行而单调递增。
*   **优点**: 适合处理稀疏数据，因为更新不频繁的参数会获得更大的学习率。
*   **缺点**: 分母不断累加，导致学习率最终会变得极小，使得模型在训练后期学习能力几乎停滞。

#### 1.2.2 RMSProp

*   **思想**: 解决 Adagrad 学习率过早衰减的问题。它不再累积所有历史梯度，而是使用**指数移动平均 (Exponential Moving Average)** 来估计近期的梯度平方大小。
*   **公式解读**:
    $$
    \sigma_t = \sqrt{\alpha (\sigma_{t-1})^2 + (1-\alpha)(g_t)^2}
    $$
    *   $\alpha$ 是衰减率（如 0.9），它控制了历史梯度信息被“遗忘”的速度。
    *   这样，分母 $\sigma_t$ 不再是单调递增的，而是反映了最近一段时间的梯度大小。
*   **优点**: 比 Adagrad 更鲁棒，是目前非常主流的优化器之一。

#### 1.2.3 Momentum (动量)

*   **思想**: 模拟物理世界中的“惯性”。参数更新的方向不仅取决于当前的梯度，还取决于上一步的更新方向。
*   **公式解读**:
    $$
    v^t = \lambda v^{t-1} - \eta \nabla L(\theta^{t-1}) \\
    \theta^t = \theta^{t-1} + v^t
    $$
    *   $v^t$ 是当前的“速度”或“动量”，$\lambda$ 是动量因子（如 0.9）。
    *   $v^t$ 是对过去所有梯度进行指数加权平均的结果，如笔记中推导的 `v^t = -η Σ...` 所示。
*   **效果**:
    1.  **加速收敛**: 如果当前梯度方向与历史方向一致，动量会加速更新。
    2.  **减少震荡**: 在梯度方向变化剧烈的“峡谷”地带，动量可以平滑更新路径，帮助冲出局部震荡。

#### 1.2.4 Adam (Adaptive Moment Estimation) - **现代神经网络的默认选择**

*   **思想**: **集大成者，RMSProp + Momentum**。
    1.  像 **Momentum** 一样，使用指数移动平均来累积**梯度的一阶矩（均值）**。
    2.  像 **RMSProp** 一样，使用指数移动平均来累积**梯度的二阶矩（平方的均值）**。
*   **优点**: 结合了两者的优点，既有自适应学习率的能力，又有动量的加速效果。在绝大多数情况下，Adam 都能取得非常好的、稳定的结果，是目前最常用的优化器。

---

## Part 2: 提升测试集表现 (Improve Results on Test Data)

这部分的目标是解决**过拟合 (Overfitting)** 的问题，即模型在训练集上表现很好，但在未见过的数据（测试集）上表现很差。这说明模型学到了训练数据中的噪声和特例，而不是通用的规律。

### 2.1 提前终止 (Early Stopping)

*   **思想**: 防止模型在训练集上“过度学习”。
*   **操作**:
    1.  将原始训练数据分为**训练集 (training set)** 和**验证集 (validation set)**。
    2.  模型在训练集上进行训练。
    3.  每隔一段时间（例如一个 epoch），在验证集上评估模型的性能（如损失或准确率）。
    4.  **当验证集上的性能不再提升，甚至开始下降时，就停止训练**。
    5.  保存验证集性能最好时的模型参数作为最终模型。
*   **优点**: 非常简单、直观且有效。

### 2.2 正则化 (Regularization)

*   **思想**: 通过在损失函数中加入一个**惩罚项**，来限制模型参数的复杂度，从而防止过拟合。
*   **核心**: 我们希望找到一组权重，它不仅能最小化原始损失，而且权重值本身也尽可能小。

#### 2.2.1 L2 正则化 (权重衰减, Weight Decay)

*   **公式**:
    $$
    L_{new}(\theta) = L_{original}(\theta) + \frac{\lambda}{2} ||w||_2^2 = L_{original}(w) + \frac{\lambda}{2} \sum_i w_i^2
    $$
*   **更新规则解读**:
    $$
    w^{t+1} \leftarrow (1 - \eta \lambda) w^t - \eta \frac{\partial L_{original}}{\partial w}
    $$
    *   在每次更新权重时，会先将权重 $w^t$ 乘以一个小于 1 的系数 $(1 - \eta \lambda)$，这相当于让权重向 0 **衰减 (decay)** 一步，然后再进行常规的梯度更新。
*   **效果**: 倾向于让所有权重都变得很小，但不为零，使得模型对输入的变化不那么敏感，更平滑。

#### 2.2.2 L1 正则化

*   **公式**:
    $$
    L_{new}(\theta) = L_{original}(\theta) + \lambda ||w||_1 = L_{original}(w) + \lambda \sum_i |w_i|
    $$
*   **更新规则解读**:
    $$
    w^{t+1} \leftarrow w^t - \eta \frac{\partial L_{original}}{\partial w} - \eta \lambda \text{sgn}(w)
    $$
    *   每次更新时，权重会朝着与自身符号相反的方向移动一个固定的步长 $\eta \lambda$。
*   **效果**: L1 正则化倾向于产生**稀疏权重 (sparse weights)**，即让许多不重要的特征权重直接变为 0。因此，L1 正则化也常被用于**特征选择**。

### 2.3 Dropout

*   **思想**: 在训练过程中，随机地“丢弃”一部分神经元，强迫网络学习更鲁棒的特征。
*   **操作 (训练时)**:
    1.  在每次前向传播时，每个神经元以概率 $p$ (例如 50%) 被暂时“丢弃”（其输出被置为 0）。
    2.  对于每个 mini-batch，我们都会随机丢弃不同的神经元，相当于在训练一个**结构不同**的“子网络”。
    3.  反向传播时，只更新未被丢弃的神经元的权重。
*   **操作 (测试时)**:
    1.  **不进行 dropout**，使用完整的网络。
    2.  将所有权重乘以 $(1-p)$。这一步是为了“补偿”训练时丢弃的神经元。直观理解是，训练时平均只有 $(1-p)$ 的神经元在工作，而测试时所有神经元都在工作，为了保持输出的期望值一致，需要对权重进行缩放。
*   **为什么有效？(集成学习视角)**:
    *   Dropout 可以被看作是一种高效的**模型集成 (Ensemble)** 方法。
    *   训练过程相当于在训练 $2^M$ (M为神经元数量) 个共享权重的子网络。
    *   测试时的权重缩放，是一种近似计算所有这些子网络预测结果平均值的技巧。
    *   通过集成大量不同的模型，可以有效降低过拟合风险。