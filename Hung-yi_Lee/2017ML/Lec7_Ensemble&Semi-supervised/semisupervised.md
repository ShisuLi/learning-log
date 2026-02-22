# Semi-supervised Learning

- Supervised learning: $\{(x^r, \hat{y}^r)\}^R_{r=1}$
- Semi-supervised learning: $\{(x^r, \hat{y}^r)\}^R_{r=1}$, $\{x^u\}^{R+U}_{u=R+1}$ 
    - A set of unlabeled data, usually U >> R
    - Transductive learning: unlabeled data is the testing data
    - Inductive learning: unlabeled data is not the testing data

- Why semi-supervised learning?
    - Collecting data is easy, labeling data is hard
    - Unlabeled data can help to learn the data distribution

- Usually with some assumptions
    - Smoothness assumption: if two data points are close, they should have the same label
    - Cluster assumption: data points tend to form clusters, and points in the same cluster should have the same label
    - Manifold assumption: high-dimensional data lie on a low-dimensional manifold, and we can learn the manifold structure from unlabeled data

## Semi-supervised Learning for Generative Model

- Given labelled training examples: $x^r \in C_1, C_2$
    - Looking for most likely prior probability $P(C_i)$ and class-dependent probability $P(x|C_i)$
    - $P(x|C_i)$ is a Gaussian parameterized by $\mu_i$ and $\Sigma$
    - The unlabeled data $x^u$ help re-estimate  $P(C_1),\ P(C_2),\ \mu_1,\ \mu_2, \Sigma$.

- Initialization: $\theta={P(C_1), P(C_2), \mu_1, \mu_2, \Sigma}$
- Step1: E-step: compute the posterior probability for unlabeled data
    - $P(C_i|x^u) = \frac{P(x^u|C_i)P(C_i)}{P(x^u|C_1)P(C_1)+P(x^u|C_2)P(C_2)}$
- Step2: M-step: re-estimate the parameters using both labeled and unlabeled data
    - $P(C_i) = \frac{\sum_{x^r \in C_i} 1 + \sum_{x^u} P(C_i|x^u)}{R+U}$
    - $\mu_i = \frac{\sum_{x^r \in C_i} x^r + \sum_{x^u} P(C_i|x^u)x^u}{\sum_{x^r \in C_i} 1 + \sum_{x^u} P(C_i|x^u)}$
    - $\Sigma = \frac{\sum_{i=1}^2 (\sum_{x^r \in C_i} (x^r - \mu_i)(x^r - \mu_i)^T + \sum_{x^u} P(C_i|x^u)(x^u - \mu_i)(x^u - \mu_i)^T)}{R+U}$

- The algorithm iterates between E-step and M-step until convergence eventually, but the initialization influences the final result.

**Why?**

- Maximum likelihood with labelled data
    $$ \log L(\theta) = \sum_{r=1}^R \log P(x^r, \hat{y}^r | \theta)\\ P(x^r, \hat{y}^r|\theta)= P(x^r| \hat{y}^r, \theta)P(\hat{y}^r|\theta) $$
    - Closed-form solution
- Maximum likelihood with labelled and unlabelled data
    $$ \log L(\theta) = \sum_{r=1}^R \log P(x^r, \hat{y}^r | \theta) + \sum_{u=R+1}^{R+U} \log P(x^u|\theta)\\P(x^u|\theta) = P(x^u|C_1, \theta)P(C_1) + P(x^u|C_2, \theta)P(C_2) $$
    - Solved iteratively by EM algorithm

## Low-density Separation Assumption

### Self-training
- Given: labelled data set =$\{(x^r, \hat{y}^r)\}^R_{r=1}$, unlabeled data set =$\{x^u\}^{R+U}_{u=R+1}$ 
- Repeat:
    - Train a model $f^*$ on the labelled data
    - Predict the labels of unlabeled data, obtain $\{(x^u, y^u)\}_{u=R+1}^{R+U}$(pseudo labels)
    - Add the most confidently predicted unlabeled data to the labelled data set
    - How to choose the data set remains open. You can also provide a weight to each data.

- Hard label & Soft label
    - Hard label: $y^u = \arg\max_c P(y=c|x^u)$
    - Soft label: use the predicted probability as the weight for each class; not useful for deep learning

### Entropy-based Regularization

- Entropy of $y^u$: Evaluate how concentrated the distribution $y^u$ is
    $$ E(y^u)= -\sum_c P(y=c|x^u) \log P(y=c|x^u) \\ L= \sum_{r=1}^R C(y^r, \hat{y}^r) + \lambda \sum_{u=R+1}^{R+U} E(y^u) $$

### Semi-supervised SVM
- Idea: Fide a boundary that can provide the largest margin and least error.
- Enumerate all possible label assignments for unlabeled data, and choose the one that gives the maximum margin

## Smoothness Assumption

- Idea: If two data points are close, they should have the same label
    - x is not uniform
    - If $x_1$ and $x_2$ are close in a high density region, $\hat{y}_1 $ and $\hat{y}_2$ are the same(connected by a high density path)

### Cluster and then Label

- Using all the data to learn a classifier as usual.
- Cluster the data points into K clusters using both labeled and unlabeled data
- Assign the label to each cluster by majority voting using labeled data

### Graph-based Approach

- How to know x1 and x2 are close in a high density region(connected by a high density path)
- Represented the data points as a graph
- Graph representation is nature sometimes: Hyperlink of web pages, social network, citation network

**Graph Construction**
- Define the similarity $s(x_i, x_j)$ between $x_i$ and $x_j$
- Add edge:
    - K Nearest Neighbor: connect $x_i$ to its K nearest neighbors
    - $\varepsilon$-neighborhood: connect $x_i$ to all points within distance
- Edge weight is proportional to $s(x_i, x_j)$
    Gaussian Radial Basis Function (RBF) kernel: $s(x_i, x_j) = \exp(-\frac{\|x_i - x_j\|^2}{2\sigma^2})$

- The labelled data influence their neighbors, propagating the labels through the graph

- Define the smoothness of the labels on the graph
    $$ S=\frac{1}{2}\sum_{i,j} s(x_i, x_j) (y_i - y_j)^2 $$
    for all data (no matter labelled or not), smaller means smoother
    $$ S=Y^TLY $$
    where Y is the (R+U)-dim vector of labels for all data, L is the (R+U) x (R+U) graph Laplacian matrix, defined as L=D-W, D is the degree matrix, W is the adjacency matrix
    - $Y^TLY$ depends on network parameters
    $$ L=\sum_{r=1}^R C(y^r, \hat{y}^r)+\lambda S $$
    $\lambda S$ as the regularization term

## Better Representation

- Find the latent factors behind the observation
- The latent factors (usually simpler) are better representations


好的，这是一份对您提供的半监督学习 (Semi-supervised Learning, SSL) 笔记的全面、详细的扩展与讲解。笔记的结构非常清晰，我将在此基础上进行深入剖析，补充背景知识、直观解释和更深层次的关联。

---

## 1. 半监督学习：核心思想与动机

### 1.1 定义与符号

*   **监督学习 (Supervised Learning)**:
    我们拥有一组完全标注好的数据，记为 $\{(x^r, \hat{y}^r)\}_{r=1}^R$。其中 $x^r$ 是第 $r$ 个样本的特征，$\hat{y}^r$ 是其对应的真实标签，$R$ 是有标签数据的总量。

*   **半监督学习 (Semi-supervised Learning)**:
    我们同时拥有两部分数据：
    1.  一小部分**有标签数据**: $\{(x^r, \hat{y}^r)\}_{r=1}^R$
    2.  一大堆**无标签数据**: $\{x^u\}_{u=R+1}^{R+U}$
    通常情况下，无标签数据的数量远大于有标签数据，即 $U \gg R$。半监督学习的目标就是**同时利用这两部分数据来构建一个性能更好的模型**。

### 1.2 两种学习范式

*   **直推式学习 (Transductive Learning)**:
    *   **核心假设**: 我们想要预测的**测试数据是已知的**，它就是我们的无标签数据集 $\{x^u\}$。
    *   **目标**: 只为这批特定的无标签数据预测标签，而不构建一个能泛化到全新未知数据的通用模型。
    *   **例子**: 在一个社交网络中，已知部分用户的属性，预测网络中其他所有用户的属性。

*   **归纳式学习 (Inductive Learning)**:
    *   **核心假设**: 无标签数据 $\{x^u\}$ **不是**最终的测试数据，它只是用来帮助我们学习数据内在结构的辅助信息。
    *   **目标**: 构建一个可以泛化到**任何**未来新样本的通用模型。
    *   **这是目前深度学习领域中半监督学习的主流范式**。

### 1.3 为什么半监督学习有效？

*   **现实困境**: 在现实世界中，获取海量数据（如图片、文本）通常很容易，但为这些数据打上精确的标签（如标注图片中的物体、翻译文本）却非常昂贵和耗时。
*   **无标签数据的价值**: "一个人的命运，当然要靠自我奋斗，但也要考虑到历史的进程"。模型的性能不仅取决于有标签数据的“指导”，也取决于对整个数据“世界”分布的理解。无标签数据虽然没有直接告诉我们“答案”，但它揭示了**数据本身的内在结构、分布和流形**。
*   **核心思想**: **利用无标签数据来更好地理解特征空间，从而帮助模型做出更准确的决策**。如下图所示，仅凭两个有标签的点，决策边界可能画在任何地方。但如果加入了大量无标签数据，我们会发现数据其实分成了两个清晰的簇，决策边界应该从它们之间的低密度区域穿过。

    

### 1.4 半监督学习的三大核心假设

半监督学习的有效性建立在一些基本假设之上，这些假设将无标签数据的分布信息与分类任务关联起来。

1.  **平滑假设 (Smoothness Assumption)**:
    *   **内容**: 如果两个样本 $x_1$ 和 $x_2$ 在特征空间中“很近”，那么它们的标签 $\hat{y}_1$ 和 $\hat{y}_2$ 应该相同。
    *   **扩展理解**: 这里的“近”不仅仅指欧氏距离近，更重要的是指它们可以通过一个**高密度区域**连接起来。

2.  **聚类假设 (Cluster Assumption)**:
    *   **内容**: 数据倾向于形成不同的簇（cluster）。同一个簇内的所有样本应该拥有相同的标签。
    *   **推论**: 一个好的决策边界应该穿过**低密度区域**，以避免将一个紧密的簇切开。这也被称为**低密度分离 (Low-density Separation)**。

3.  **流形假设 (Manifold Assumption)**:
    *   **内容**: 高维数据实际上位于一个嵌入在高维空间中的**低维流形 (manifold)** 上。
    *   **扩展理解**: 我们可以认为数据是由少数几个潜在因素（latent factors）生成的，这些因素构成了低维流形。半监督学习可以利用所有数据来学习这个流形的结构，从而使得分类任务变得更简单。

---

## 2. 基于生成模型的半监督学习

这是半监督学习最经典的方法之一，它试图直接对数据的生成过程进行建模。

### 2.1 算法流程 (EM算法)

假设我们正在做一个二分类问题 ($C_1, C_2$)，并且我们假设每个类别的数据都服从一个高斯分布 $P(x|C_i) = \mathcal{N}(x | \mu_i, \Sigma)$。我们的目标是利用所有数据来估计模型的参数 $\theta = \{P(C_1), P(C_2), \mu_1, \mu_2, \Sigma\}$。

1.  **初始化 (Initialization)**:
    *   首先，仅用少量有标签数据来粗略估计一版初始参数 $\theta^0$。例如，计算 $C_1$ 类有标签数据的均值作为 $\mu_1^0$ 的初始值。
    *   这个初始化的好坏会显著影响最终结果。

2.  **重复以下两步直至收敛**:

    *   **E-Step (Expectation Step)**:
        *   **目标**: 根据当前的模型参数 $\theta^t$，为**所有无标签数据** $x^u$ 计算它属于每个类别的“期望”或后验概率。
        *   **公式**: 使用贝叶斯定理计算 $x^u$ 属于 $C_i$ 的概率：
            $$
            P(C_i | x^u; \theta^t) = \frac{P(x^u | C_i; \theta^t) P(C_i; \theta^t)}{\sum_{k=1}^2 P(x^u | C_k; \theta^t) P(C_k; \theta^t)}
            $$
        *   这一步相当于为无标签数据打上了“软标签” (soft label)。

    *   **M-Step (Maximization Step)**:
        *   **目标**: 使用**全部数据**（有标签数据 + 打上软标签的无标签数据）来重新估计（最大化似然）模型参数，得到新一版的 $\theta^{t+1}$。
        *   **公式解读**:
            *   **先验概率 $P(C_i)$**: 分子是有标签数据中属于 $C_i$ 的数量，加上所有无标签数据属于 $C_i$ 的概率之和。分母是总样本数。
                $$ P(C_i) = \frac{N_i^R + \sum_{x^u} P(C_i|x^u)}{R+U} $$
            *   **均值 $\mu_i$**: 分子是 $C_i$ 类有标签数据的特征总和，加上每个无标签数据 $x^u$ 按其属于 $C_i$ 的概率 $P(C_i|x^u)$ 加权后的特征总和。分母是 $C_i$ 的总“有效”数量。
                $$ \mu_i = \frac{\sum_{x^r \in C_i} x^r + \sum_{x^u} P(C_i|x^u)x^u}{N_i^R + \sum_{x^u} P(C_i|x^u)} $$
            *   **协方差 $\Sigma$**: 类似地，将每个样本（有标签或无标签）的协方差矩阵按其类别归属（有标签为1，无标签为后验概率）进行加权平均。

### 2.2 为什么这样做？(最大似然视角)

*   **仅有标签数据**: 损失函数（负对数似然）为 $\sum_{r=1}^R \log P(x^r, \hat{y}^r)$。这个问题有闭式解，可以直接计算出最优参数。
*   **加入无标签数据**: 损失函数变为：
    $$
    \log L(\theta) = \sum_{r=1}^R \log P(x^r, \hat{y}^r|\theta) + \sum_{u=R+1}^{R+U} \log P(x^u|\theta)
    $$
    其中，无标签数据的似然 $P(x^u|\theta)$ 是一个混合项：$P(x^u|\theta) = \sum_{k=1}^K P(x^u|C_k, \theta)P(C_k|\theta)$。这个“和的对数”形式导致整个问题没有闭式解。
*   **EM算法的角色**: EM (Expectation-Maximization) 算法是一种迭代求解这类含有隐变量（这里是无标签数据的真实类别）的最大似然问题的标准方法。E-step 估计隐变量的期望，M-step 在此期望下最大化似然函数。

---

## 3. 基于低密度分离假设的方法

这类方法的核心是**决策边界应该穿过数据稀疏的区域**。

### 3.1 自训练 (Self-training)

这是最简单、最直观的半监督方法之一。

*   **流程**:
    1.  **训练**: 仅用有标签数据训练一个初始模型 $f^*$。
    2.  **伪标签 (Pseudo-Labeling)**: 用 $f^*$ 去预测所有无标签数据 $\{x^u\}$ 的标签，得到一组“伪标签” $\{y^u\}$。
    3.  **筛选**: 从这批伪标签数据中，挑选出模型**最自信**的样本（例如，预测概率最高的那些）。
    4.  **扩充**: 将这些高置信度的样本及其伪标签加入到有标签数据集中。
    5.  **重复**: 回到第一步，用扩充后的数据集重新训练模型。
*   **硬标签 vs 软标签**:
    *   **硬标签**: 直接将概率最高的类别作为伪标签（$y^u=1$ 或 $0$）。这是最常见的做法。
    *   **软标签**: 直接使用模型输出的概率分布（如 `[0.9, 0.1]`）作为标签。在深度学习中，因为模型本身就是用交叉熵训练的，直接使用软标签进行 self-training 意义不大，模型无法从自己的预测中学到新东西。
*   **风险**: Self-training 的最大风险是**错误累积**。如果模型一开始就给某个样本打上了错误的伪标签，这个错误会在后续的训练中被不断放大和巩固，可能导致模型性能恶化。

### 3.2 基于熵的正则化 (Entropy-based Regularization)

*   **思想**: 如果我们相信决策边界应该在低密度区域，那么对于一个无标签样本 $x^u$，模型的预测分布 $P(y|x^u)$ 应该是**低熵的**，即非常确定的（例如 `[0.99, 0.01]`），而不是模棱两可的（例如 `[0.5, 0.5]`）。
*   **实现**: 在原始的监督损失之外，增加一个正则化项，该项用于**最小化无标签数据预测结果的熵**。
    $$
    L = \underbrace{\sum_{r=1}^R C(y^r, \hat{y}^r)}_{\text{监督损失}} + \lambda \underbrace{\sum_{u=R+1}^{R+U} E(y^u)}_{\text{熵正则化项}}
    $$
    其中 $E(y^u) = -\sum_c P(y=c|x^u) \log P(y=c|x^u)$。通过梯度下降最小化这个总损失，模型会被迫对无标签数据做出更确定的预测。

### 3.3 半监督支持向量机 (Semi-supervised SVM, S3VM)

*   **思想**: SVM 的目标是找到一个能最大化间隔 (margin) 的决策边界。S3VM 将这个思想扩展到半监督场景：寻找一个决策边界，它不仅能正确分类有标签数据，而且能以**最大间隔**穿过无标签数据，同时尽量将无标签数据分到间隔的两侧。
*   **挑战**: 这是一个非凸优化问题，计算成本非常高。需要枚举无标签数据所有可能的标签组合，然后为每种组合训练一个 SVM，最后选出间隔最大的那个。实践中通常使用近似算法求解。

---

## 4. 基于平滑假设的方法

这类方法的核心是**特征相似的样本，其标签也应该相似**。图方法是实现这一思想的有力工具。

### 4.1 思路一：先聚类后标注 (Cluster and then Label)

*   **流程**:
    1.  **聚类**: 忽略所有标签，将**全部数据**（有标签和无标签）进行聚类（如 K-Means）。
    2.  **标注**: 对每个聚类，使用该簇内的有标签数据进行“投票”，将得票最多的标签赋给该簇内的所有样本。
*   **局限**: 效果高度依赖于聚类算法的质量以及数据本身的簇结构是否清晰。

### 4.2 思路二：基于图的方法 (Graph-based Approach)

这种方法将平滑假设表达得更加精致和灵活。

*   **核心步骤**:
    1.  **图构建 (Graph Construction)**:
        *   将每一个数据样本（无论有无标签）都看作图中的一个**节点**。
        *   在节点之间添加**边**来表示它们的“相似性”。常用的建边策略有：
            *   **K近邻 (K-NN)**: 每个节点只与它最相似的 K 个节点相连。
            *   **$\epsilon$-邻域 ($\epsilon$-neighborhood)**: 每个节点与和它距离在 $\epsilon$ 之内的所有节点相连。
        *   为每条边赋予**权重** $s(x_i, x_j)$，权重大小与相似度成正比。常用的相似度函数是**高斯径向基函数 (RBF)**:
            $$
            s(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
            $$

    2.  **标签传播 (Label Propagation)**:
        *   图构建好后，有标签节点的标签就像“源头”一样，会通过带权的边“传播”或“扩散”到它们的邻居，邻居再传给邻居的邻居，直至整个图达到稳定状态。
        *   直观上，两个节点如果通过一条由很多短且权重高的边构成的路径连接，那么它们就很可能拥有相同的标签。

    3.  **定义图平滑度 (Smoothness on the Graph)**:
        *   我们如何用数学语言描述“标签在图上是平滑的”？一个好的定义是：**相邻节点的标签差异应该很小**。
        *   平滑度函数 $S$ 可以定义为所有相连节点标签差异的加权平方和：
            $$
            S = \frac{1}{2}\sum_{i,j} s(x_i, x_j) (y_i - y_j)^2
            $$
            $S$ 越小，说明标签分布越平滑。
        *   这个式子可以被优雅地写成矩阵形式：$S = Y^T L Y$，其中：
            *   $Y$ 是一个包含所有数据（有标签和无标签）标签的向量。
            *   $L$ 是**图拉普拉斯矩阵 (Graph Laplacian)**，定义为 $L=D-W$。$W$ 是图的邻接权重矩阵，$D$ 是对角度的矩阵 ($D_{ii} = \sum_j W_{ij}$)。拉普拉斯矩阵是图论和谱分析中的核心工具。

    4.  **整合为损失函数**:
        *   将图的平滑度作为一个**正则化项**加入到总损失函数中。
            $$
            L_{total} = \underbrace{\sum_{r=1}^R C(y^r, \hat{y}^r)}_{\text{监督损失}} + \lambda \underbrace{S}_{\text{图平滑度正则项}}
            $$
        *   通过最小化这个总损失，模型不仅要努力拟合有标签数据，还要保证其在整个图上的预测结果是平滑的，从而将在有标签数据上学到的知识泛化到无标签数据上。

---

## 5. 基于更优表示的方法 (Better Representation)

*   **思想**: 这个思路与流形假设紧密相关。它认为我们观察到的高维数据（如像素、词向量）背后，其实是由一些更简单、更本质的**潜在因素 (latent factors)** 控制的。如果我们能学习到一个好的**表示 (representation)**，将原始数据映射到由这些潜在因素构成的特征空间，那么后续的分类任务就会变得异常简单。
*   **实现**:
    *   **自编码器 (Autoencoder)**: 这是一个经典的无监督表示学习方法。可以用**全部数据**（有标签+无标签）来训练一个自编码器。其目标是学习一个编码器 (Encoder) 和一个解码器 (Decoder)，使得输入数据经过编码再解码后能尽可能地被还原。
    *   训练完成后，这个**编码器**就学会了如何从原始数据中提取有用的、压缩的表示。
    *   然后，我们可以用这个预训练好的编码器来提取所有有标签数据的特征表示，再用这些“更好”的特征来训练一个简单的分类器（如逻辑回归）。
    *   现代半监督学习方法，如**对比学习 (Contrastive Learning)**，就是这一思想的延伸和发扬，它们通过精心设计的代理任务（pretext task）在无标签数据上学习强大的特征表示。

好的，我们来深入、详细地讲解高斯径向基函数 (RBF) 和图拉普拉斯矩阵 (Graph Laplacian)。这两个工具虽然源于不同的数学领域，但在现代机器学习，尤其是处理非结构化数据和图数据时，扮演着至关重要的角色。

---

## 1. 高斯径向基函数 (Gaussian Radial Basis Function, RBF)

高斯 RBF 函数是一种应用极其广泛的**核函数 (Kernel Function)**，其核心作用是**衡量两个点之间的“相似度”**。

### 1.1 定义与公式

对于两个 D 维空间中的数据点 $x_i$ 和 $x_j$，它们之间的高斯 RBF 相似度定义为：

$$
s(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
$$

让我们来逐项分解这个公式：

*   **$\|x_i - x_j\|^2$**: 这是两个点之间距离的**平方欧氏距离**。它是公式中唯一依赖于输入数据点的部分。
*   **径向基 (Radial Basis)**: 这个词意味着函数的值**只依赖于点与某个中心点之间的距离**。在这里，你可以把 $x_j$ 看作是中心，函数值仅取决于 $x_i$ 离它有多远，而与方向无关。
*   **$\sigma$ (Sigma)**: 这是一个**超参数**，通常称为**带宽 (bandwidth)** 或**长度尺度 (length-scale)**。它控制着函数“衰减”的速度，是 RBF 函数的灵魂所在。
    *   **小的 $\sigma$**: 函数图像会变得非常“尖锐”。只有当两个点**非常非常近**的时候，相似度才接近 1；距离稍微一远，相似度就迅速衰减到 0。这使得模型对距离非常敏感。
    *   **大的 $\sigma$**: 函数图像会变得非常“平缓”。即使两个点相距较远，它们的相似度也可能保持在一个较高的水平。这使得模型的决策边界更平滑。
*   **$\exp(\cdot)$**: 指数函数将距离映射到 $(0, 1]$ 区间内，完美地充当了“相似度”的角色。
    *   当 $x_i = x_j$ 时，距离为 0，相似度为 $\exp(0) = 1$ (最相似)。
    *   当 $\|x_i - x_j\| \to \infty$ 时，距离无限大，相似度为 $\exp(-\infty) \to 0$ (最不相似)。

### 1.2 直观理解与应用

想象一下，你在构建一个基于图的半监督学习模型：

1.  **节点**: 每个数据点都是图上的一个节点。
2.  **边的权重**: 你需要决定节点之间边的权重，以表示它们的亲疏关系。高斯 RBF 就是计算这个权重的完美工具。

你拿起两个点 $x_i$ 和 $x_j$，用 RBF 函数计算出它们的相似度 $s(x_i, x_j)$，这个值就成了连接它们之间那条边的权重。如果你对所有点对都这样做，你就构建了一个**全连接的加权图**。

在实践中，为了避免图过于稠密（计算成本高），通常会结合 K近邻 (KNN) 或 $\epsilon$-邻域策略：只在满足特定邻近条件的点之间计算 RBF 权重并连边，其他点之间没有边（或权重为0）。

**总结：RBF 的本质是一个灵活的、可调节的“距离到相似度”的转换器。通过调整 $\sigma$，你可以控制“近”的定义，从而影响整个图的结构和后续的标签传播行为。**

---

## 2. 图拉普拉斯矩阵 (Graph Laplacian)

图拉普拉斯矩阵是图论与线性代数之间的桥梁，它是一个能**描述图的结构和连通性**的矩阵。在机器学习中，它最核心的作用是**定义和计算图上的“平滑度”或“总变差”**。

### 2.1 定义

对于一个包含 $N$ 个节点的图，其图拉普拉斯矩阵 $L$ 是一个 $N \times N$ 的矩阵，定义为：

$$
L = D - W
$$

*   **$W$ (邻接权重矩阵, Adjacency Matrix)**:
    *   这是一个 $N \times N$ 的矩阵。
    *   $W_{ij}$ 表示节点 $i$ 和节点 $j$ 之间边的权重。如果两个节点间没有边，则 $W_{ij}=0$。
    *   这个权重通常就是用我们上面讨论的 RBF 函数计算得到的 $s(x_i, x_j)$。

*   **$D$ (度矩阵, Degree Matrix)**:
    *   这是一个 $N \times N$ 的**对角矩阵**（只有主对角线有非零值）。
    *   对角线上的元素 $D_{ii}$ 是节点 $i$ 的“度”，即与节点 $i$ 相连的所有边的**权重之和**。
    *   $D_{ii} = \sum_{j=1}^{N} W_{ij}$

**举个例子**：
假设一个3个节点的简单图，权重如下：
$W_{12}=3, W_{13}=1, W_{23}=2$
那么：
$W = \begin{pmatrix} 0 & 3 & 1 \\ 3 & 0 & 2 \\ 1 & 2 & 0 \end{pmatrix}$

$D_{11} = 3+1=4$
$D_{22} = 3+2=5$
$D_{33} = 1+2=3$
$D = \begin{pmatrix} 4 & 0 & 0 \\ 0 & 5 & 0 \\ 0 & 0 & 3 \end{pmatrix}$

$L = D - W = \begin{pmatrix} 4 & -3 & -1 \\ -3 & 5 & -2 \\ -1 & -2 & 3 \end{pmatrix}$

### 2.2 拉普拉斯矩阵的核心作用：衡量“平滑度”

现在到了最关键的部分。假设我们给图上的每个节点 $i$ 赋予一个值 $y_i$（比如，模型的预测标签或概率）。我们可以把所有节点的 $y$ 值组成一个向量 $Y = [y_1, y_2, \dots, y_N]^T$。

我们如何衡量这个标签向量 $Y$ 在图上的**平滑程度**？一个直观的想法是：**如果相邻节点的标签差异很小，那么它就是平滑的**。

我们可以定义一个“总变差”或“不平滑度” $S$：
$$
S = \frac{1}{2} \sum_{i,j} W_{ij} (y_i - y_j)^2
$$
这个公式的含义是：遍历图中所有相连的节点对 $(i,j)$，计算它们标签值的差的平方，再乘以它们之间的连接权重 $W_{ij}$。如果连接很紧密（$W_{ij}$大）的两个节点标签差异很大（$(y_i - y_j)^2$大），那么总的不平滑度 $S$ 就会很大。

**神奇之处在于，这个求和公式可以被完美地写成一个简洁的二次型**：

$$
S = Y^T L Y
$$

*   **为什么？(推导)**
    $Y^T L Y = Y^T(D-W)Y = Y^T D Y - Y^T W Y$
    $= \sum_i D_{ii} y_i^2 - \sum_{i,j} W_{ij} y_i y_j$
    把 $D_{ii} = \sum_j W_{ij}$ 代入：
    $= \sum_i \left( \sum_j W_{ij} \right) y_i^2 - \sum_{i,j} W_{ij} y_i y_j$
    $= \frac{1}{2} \left( \sum_{i,j} W_{ij} y_i^2 + \sum_{i,j} W_{ji} y_j^2 \right) - \sum_{i,j} W_{ij} y_i y_j$  (因为 $W_{ij}=W_{ji}$，所以一项可以拆成两项)
    $= \frac{1}{2} \sum_{i,j} W_{ij} (y_i^2 - 2y_i y_j + y_j^2)$
    $= \frac{1}{2} \sum_{i,j} W_{ij} (y_i - y_j)^2$
    推导完成！

### 2.3 在半监督学习中的应用

在半监督学习中，这个平滑度项 $Y^TLY$ 成了一个强大的**正则化项**。

1.  我们构建一个包含所有数据（有标签和无标签）的图，并计算出其拉普拉斯矩阵 $L$。
2.  $Y$ 向量包含了模型对所有数据的预测。
3.  我们将总损失函数定义为：
    $$
    L_{total} = \underbrace{\text{Loss}_{\text{supervised}}(Y_{\text{labeled}}, \hat{Y}_{\text{labeled}})}_{\text{有标签数据的监督损失}} + \lambda \underbrace{Y^T L Y}_{\text{图平滑度正则项}}
    $$
4.  通过梯度下降最小化这个总损失，模型在学习的过程中会面临一种“拉扯”：
    *   监督损失项会把有标签数据的预测值 $Y_{\text{labeled}}$ “拉”向它们的真实标签 $\hat{Y}_{\text{labeled}}$。
    *   图平滑度正则项会把整个图的预测 $Y$ “拉”得更平滑，即让通过强连接（高权重边）相连的节点的预测值趋于一致。

最终的结果是，有标签数据上的“知识”会通过图的结构自然地“传播”到无标签数据上，使得模型的决策边界能够尊重数据的内在流形结构。

**总结：图拉普拉斯矩阵 $L$ 是一个神奇的算子，它将一个图的拓扑结构编码进一个矩阵中。当作用于一个与图节点相关的向量 $Y$ 时（通过 $Y^TLY$），它能高效地计算出这个向量在图上的平滑程度，从而成为连接图结构与机器学习损失函数的完美桥梁。**

好的，我们来深入探讨一下当前在深度学习领域**更现代、更前沿、也更实用**的半监督学习（SSL）方法。

这些方法与传统的SSL方法（如生成模型、图方法）相比，最大的区别在于它们被专门设计用来与**深度神经网络**无缝结合，并且能够在图像、文本等高维数据上取得惊人的效果。它们的核心思想，几乎都围绕着一个共同的目标：**利用无标签数据进行有效的表示学习 (Representation Learning)**。

现代SSL方法的主流范式可以概括为 **一致性正则化 (Consistency Regularization)** 和 **伪标签 (Pseudo-Labeling)** 的巧妙结合，并在此基础上衍生出各种变体。

---

### 核心思想：一致性正则化 (Consistency Regularization)

这是现代半监督学习的基石。其核心假设是：

> **对于一个无标签样本，如果我们对它施加一些微小但合理的扰动，模型的预测结果不应该发生显著变化。**

这个思想非常直观：一只猫的图片，无论它被轻微地旋转、裁剪、调整亮度，或者加入一点点噪声，它依然是一只猫，模型也应该始终坚信它是一只猫。

*   **如何实现？**
    1.  取一个无标签样本 $x^u$。
    2.  对 $x^u$ 进行两种不同的、随机的数据增强（Data Augmentation），得到两个“视角”：$x_1^u = \text{Aug}_1(x^u)$ 和 $x_2^u = \text{Aug}_2(x^u)$。
    3.  将这两个增强后的样本分别输入模型，得到两个预测分布 $p_1 = f(x_1^u)$ 和 $p_2 = f(x_2^u)$。
    4.  **强制这两个分布尽可能相似**。这通过在损失函数中加入一个衡量两个分布差异的项（如均方误差或KL散度）来实现。

    $$
    L_{unsupervised} = \text{Distance}(p_1, p_2)
    $$

这种方法强迫模型学习那些对于数据增强“不变”的本质特征，从而学到更鲁棒、更具泛化能力的表示。

---

### 前沿实用的半监督学习方法

下面介绍几种在学术界和工业界都产生巨大影响的代表性方法。

#### 1. MixMatch: 一致性正则化与数据增强的极致融合

MixMatch 是 Google 提出的一个里程碑式的工作，它将多种技术优雅地整合在一起，效果显著。

**核心步骤 (针对一个 mini-batch)**:

1.  **数据增强**: 对每一个有标签样本 $x^r$ 只做一次标准的增强。对每一个无标签样本 $x^u$，进行 **K 次**不同的随机增强，得到 $\{x_{u,1}, x_{u,2}, \dots, x_{u,K}\}$。

2.  **伪标签生成与锐化 (Label Guessing & Sharpening)**:
    *   将 K 个增强后的无标签样本输入模型，得到 K 个预测分布。
    *   将这 K 个预测分布**取平均**，得到一个更稳定、更可靠的“平均预测” $\bar{p}_u$。
    *   对这个平均预测进行**锐化 (Sharpening)** 操作。这是一个关键步骤，目的是让伪标签的分布熵更低（更确定）。常用的锐化函数是调整“温度”T：
        $$
        \text{Sharpen}(p, T)_i = \frac{p_i^{1/T}}{\sum_{j=1}^C p_j^{1/T}}
        $$
        当 $T \to 0$ 时，这个分布会趋近于 one-hot 形式，即变得非常“尖锐”。这一步体现了**低密度分离**的思想。

3.  **MixUp**: 这是 MixMatch 的另一个精髓。它将有标签数据和无标签数据（携带伪标签）进行“混合”。
    *   将所有有标签样本和所有（经过K次增强的）无标签样本混合在一起，形成一个大的数据集 W。
    *   从 W 中随机抽取样本对进行 MixUp 操作：
        $$
        x' = \lambda x_1 + (1-\lambda)x_2 \\
        y' = \lambda y_1 + (1-\lambda)y_2
        $$
        其中 $\lambda$ 是从 Beta 分布中采样的混合系数。
    *   这样，模型在训练时看到的是“混合”后的人造样本，这是一种非常强大的数据增强和正则化手段，可以鼓励模型在样本之间进行平滑的线性插值。

4.  **计算损失**:
    *   对于混合后的有标签数据，计算标准的交叉熵损失。
    *   对于混合后的无标签数据，计算**均方误差 (MSE)** 损失，衡量模型预测与（混合后的）伪标签之间的差异。

**总结**: MixMatch 通过**平均预测**和**锐化**生成高质量的伪标签，再通过 **MixUp** 这种强大的数据增强方式来同时利用有标签和无标签数据，取得了非常好的效果。

#### 2. FixMatch: 伪标签与一致性正则化的简化与强化

FixMatch 是 MixMatch 的一个简化版，但效果甚至更好，也更容易实现，是目前非常流行的基线方法。

**核心思想**: **使用弱增强生成伪标签，使用强增强来学习匹配这个伪标签。**

**核心步骤 (针对一个无标签样本 $x^u$)**:

1.  **生成两种增强**:
    *   **弱增强 (Weak Augmentation)**: 只进行简单的、标准的增强，如随机翻转和裁剪。得到 $x_w^u$。
    *   **强增强 (Strong Augmentation)**: 使用非常激进的增强策略，如 [RandAugment](https://arxiv.org/abs/1909.13719) 或 [CTA](https://arxiv.org/abs/1905.04961)，这些方法会进行大幅度的色彩、对比度、几何变换。得到 $x_s^u$。

2.  **生成伪标签**:
    *   将**弱增强**后的样本 $x_w^u$ 输入模型，得到预测分布 $p_w = f(x_w^u)$。
    *   **如果**这个预测的最大置信度超过一个预设的阈值 $\tau$（例如 0.95），那么就接受它的 one-hot 形式作为伪标签 $\hat{y}^u$。否则，就忽略这个样本。这一步被称为**置信度阈值化 (Confidence Thresholding)**。

3.  **计算一致性损失**:
    *   将**强增强**后的样本 $x_s^u$ 输入模型，得到预测分布 $p_s = f(x_s^u)$。
    *   计算 $p_s$ 与我们刚刚生成的伪标签 $\hat{y}^u$ 之间的交叉熵损失。
        $$
        L_{unsupervised} = \mathbb{I}(\max(p_w) > \tau) \cdot H(\hat{y}^u, p_s)
        $$
        其中 $\mathbb{I}(\cdot)$ 是指示函数。

**总结**: FixMatch 的设计非常优雅和高效。它认为弱增强的样本更容易被正确分类，因此用它来生成可靠的伪标签。然后，它强迫模型在看到一个被“面目全非”的强增强版本后，依然能认出它，并给出与弱增强版本一致的（伪）标签。这种“从易到难”的学习范式非常有效。

#### 3. SimCLR & MoCo: 自监督对比学习 (Self-Supervised Contrastive Learning)

虽然这类方法最初是为纯粹的**自监督学习**（即完全没有标签）设计的，但它们提供了一种极其强大的在无标签数据上进行**预训练 (Pre-training)** 的范式，可以被完美地整合进半监督流程中。

**核心思想**: **拉近“相似”的样本，推开“不相似”的样本。**

**核心步骤 (以 SimCLR 为例)**:

1.  **生成正样本对**:
    *   取一个无标签样本 $x^u$。
    *   对其进行两次**不同的随机强增强**，得到一个**正样本对 (positive pair)** $(x_i, x_j)$。我们认为这两个样本在语义上是相同的。

2.  **生成负样本**:
    *   在一个 mini-batch 中的其他所有样本（经过增强后）都被视为这对正样本的**负样本 (negative samples)**。

3.  **表示学习**:
    *   将所有增强后的样本输入一个编码器网络 (Encoder)，得到它们的表示向量 $z$。
    *   再将表示向量输入一个小的投影头网络 (Projection Head)，得到用于计算损失的向量 $h$。

4.  **计算对比损失 (Contrastive Loss)**:
    *   损失函数的目标是：在表示空间中，最大化正样本对 $(h_i, h_j)$ 之间的**相似度**，同时最小化 $h_i$ 与所有负样本之间的相似度。
    *   常用的损失函数是 **NT-Xent (Normalized Temperature-scaled Cross-Entropy)**。

**如何用于半监督学习？(两阶段法)**

1.  **预训练阶段**: 使用**所有**数据（有标签+无标签），但忽略所有标签，运行对比学习算法（如 SimCLR 或 MoCo），训练一个强大的特征编码器。
2.  **微调阶段 (Fine-tuning)**:
    *   “冻结”编码器的参数（或以很小的学习率进行微调）。
    *   在编码器之上加一个简单的线性分类头。
    *   仅用**少量有标签数据**来训练这个分类头。

由于预训练阶段已经让模型学会了如何从海量无标签数据中提取强大的、具有语义区分性的特征，因此第二阶段的微调通常只需要很少的标签就能达到非常高的性能。

---

### 总结与展望

| 方法 | 核心机制 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| **MixMatch** | K次增强平均 + 锐化 + MixUp | 效果好，思想全面，对数据增强不敏感 | 流程相对复杂，参数较多 |
| **FixMatch** | 弱增强伪标签 + 强增强学习 + 置信度阈值 | 简单、高效、性能强大，成为事实上的标准基线 | 对强数据增强的质量依赖较大 |
| **SimCLR/MoCo** (用于SSL) | 对比学习预训练 + 微调 | 能从海量无标签数据中学到极强的通用特征表示 | 两阶段训练流程，需要较大的 batch size 或 memory bank |

**当前趋势**:

*   **伪标签与一致性正则化的统治地位**: FixMatch 的成功证明了这是一个非常强大且简洁的范式。后续的很多工作，如 **FlexMatch**、**Dash** 等，都是在 FixMatch 的基础上进行改进（例如，动态调整置信度阈值）。
*   **对比学习的崛起**: 对比学习作为一种通用的表示学习框架，其潜力巨大。将对比学习与一致性正则化更深度地结合是目前的一个研究热点。
*   **超越分类**: 将这些强大的半监督思想应用到更复杂的任务上，如目标检测、语义分割、自然语言处理等，是目前前沿研究的重要方向。

总而言之，现代半监督学习已经从利用无标签数据“辅助”分类，转变为**以无标签数据为主导进行大规模表示学习，再用少量有标签数据进行“对齐”和“校准”**的新范式。