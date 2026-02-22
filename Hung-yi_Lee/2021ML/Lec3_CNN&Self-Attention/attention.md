# Attention
### Input
Simple: Input is a vector - Model - Scalar or Class
Sophisticated: Input is a set of vectors(may change length) - Model - Scalars or Classes

### Vector Set as Input

- One-hot Encoding 没有语义关系
- Word Embedding 有语义关系
- 声音变成一段段frame向量
- Graph is a set of vectors, consider each node as a vector. 药物分子、社交网络

### Output
Example Applications
- Each vector has a label输入输出长度相同：Sequence Labeling, POS tagging语义标注 
- The whole sequence has a label: Sentiment analysis情感分析
- Model decides the number of labels itself: seq2seq 机器翻译

### Sequence Labeling
- Fully connected layer can consider the neighbor, but how to consider the whole sequence?

### Self-Attention
- Attention is all you need发扬光大
- 每个input的vector都会输出一个output vector，每个vector都考虑整个句子
input-self attention处理完整句子-fc处理局部信息-self attention-fc-output

- Find the relevant vectors in a sequence: how to calculate
- Dot-product: wq-q, wk-k, $\alpha = q \cdot k$
- Additive: tahh(Wq+Wk)-W-alpha
$$
q^1=W^q a^1, k^1=W^k a^1, k^2=W^k a^2, 
\alpha_{1, 1} = q^1 \cdot k^1, \alpha_{1,2} = q^1 \cdot k^2
$$
$\alpha_{1, 1}, \alpha_{1, 2}, \alpha_{1, 3}, \ldots$-Softmax-$\hat{\alpha}_{1, 1}, \hat{\alpha}_{1, 2}, \hat{\alpha}_{1, 3}, \ldots$
$$
\hat{\alpha}_{1, i}=exp(\alpha_{1, i})/\sum_j exp(\alpha_{1, j})
$$

- 根据attention分数提取信息，$v^1=W^v a^1, v^2=W^v a^2, \ldots$, $b^1 = \sum_i \hat{\alpha}_{1, i} v^i$，每一个b被同时计算出来
- Wq, Wk, Wv are learned during training

### Multi-Head Attention
n_heads
b^{i,1} concat b^{i, 2}, W^O, = b^i

### Positional Encoding
- No position information in self-attention
- Each position has a unique positional encoding vector e^i
- Add the positional encoding vector to the input vector at each position
- hand-crafted, sin cos func
- can be learned from data


#### Self-attention for speech
- Truncated Self-attention 避免L*L计算量太大
#### Self-attention for Image
- An image can also be considered as a vector set
- CNN: self attention that can only attends in a receptive field; CNN is simplified self-attention
- Self-attention: CNN with learnable receptive field; Self-attention is the complex version of CNN


- Self-attention v.s. RNN: memory+input-RNN-FC-output, hard to consider early input, non-parallel

#### Self-attention for Graph
- Consider edge: only attention to connected nodes

好的，我们来将这份关于Attention机制的笔记，扩展成一份全面、细致且反映最新知识的深度解析。这份材料将不仅填充笔记中的空白，还会深入其背后的动机、数学原理、各种变体及其在不同领域的应用。

---

### **注意力机制 (Attention) - 让模型学会“聚焦”**

在深度学习的演进中，如何处理**可变长度的向量集合 (Set of Vectors)** 一直是一个核心挑战。传统的神经网络，如全连接网络（MLP）或卷积神经网络（CNN），通常假设输入是固定尺寸的。而注意力机制，尤其是自注意力（Self-Attention），为解决这个问题提供了革命性的方案。

#### **第一部分：为什么需要处理“向量集合”？**

在真实世界中，许多重要的数据天然就是一组向量的集合，其长度是可变的：

1.  **自然语言 (Text)**:
    *   **One-hot Encoding**: 早期方法，将每个词表示为一个巨大的、只有一个1其余都是0的向量。**缺点**: 向量维度灾难，且无法表达词与词之间的语义关系（如“国王”和“王后”的向量是正交的，看不出任何关联）。
    *   **词嵌入 (Word Embedding)**: 如Word2Vec, GloVe。将每个词映射到一个低维、稠密的向量空间中。在这个空间里，**语义相近的词，其向量也相近**。一句话就成了一个词向量的序列/集合。

2.  **语音 (Speech)**:
    *   一段语音信号可以被切分成一系列非常短的**帧 (Frame)**。每一帧都可以用一个向量（如MFCC特征）来表示。因此，整段语音就构成了一个帧向量的序列/集合。

3.  **图 (Graph)**:
    *   一个图由节点和边组成。我们可以将**每个节点 (Node) 都表示为一个向量**（包含节点的属性信息）。因此，整个图就可以被看作一个节点向量的集合。这在药物分子发现、社交网络分析等领域至关重要。

#### **第二部分：输入与输出的对应关系**

根据输入输出的对应关系，处理向量集合的任务可以分为三类：

1.  **N vs N (输入输出长度相同)**:
    *   **任务**: 为输入序列中的每一个向量都生成一个对应的标签或输出向量。
    *   **例子**: **序列标注 (Sequence Labeling)**，如词性标注（POS tagging），即为句子中的每个词标注其词性（名词、动词等）。

2.  **N vs 1 (输入一个序列，输出一个结果)**:
    *   **任务**: 模型需要“阅读”整个输入序列，并给出一个总体的判断。
    *   **例子**: **情感分析 (Sentiment Analysis)**，判断一整段评论是正面的还是负面的。

3.  **N vs M (输入输出长度不确定，由模型决定)**:
    *   **任务**: 模型需要理解输入序列的全部信息，并生成一个全新的、长度可能不同的序列。
    *   **例子**: **序列到序列 (Seq2Seq)** 任务，如机器翻译、文本摘要。这是最复杂的场景，也是Attention机制大放异彩的地方。

#### **第三部分：自注意力机制 (Self-Attention) - 内部关系的探索者**

在Attention机制出现之前，处理序列的主要工具是循环神经网络（RNN）。但RNN存在两大问题：1) 难以并行计算；2) 长距离依赖问题，即很难捕捉序列中相距很远的元素之间的关系。

**自注意力机制**彻底改变了这一切，它的核心思想是：**在处理序列中的某个元素时，动态地、全局地计算序列中所有其他元素对该元素的重要性（即“注意力”），然后根据这个重要性来加权聚合整个序列的信息，从而得到该元素的新表示。**

##### **Self-Attention 的运作流程**

这个过程可以被分解为三个优雅的步骤，以计算序列中第一个向量 $a^1$ 的新表示 $b^1$ 为例：

**Step 1: 生成 Query, Key, Value 三个向量**

对于输入序列中的每一个向量 $a^i$，我们都通过三个**可学习的**权重矩阵 $W^q, W^k, W^v$ 来生成对应的三个新向量：
*   **Query (查询向量) $q^i = W^q a^i$**: 代表了当前元素为了更好地理解自己，需要去“查询”或“关注”什么信息。
*   **Key (键向量) $k^i = W^k a^i$**: 代表了当前元素自身所包含的、可供其他元素“查询”的“索引”或“标签”信息。
*   **Value (值向量) $v^i = W^v a^i$**: 代表了当前元素自身所包含的实际内容。

**Step 2: 计算注意力分数 (Attention Score)**

为了计算 $a^1$ 对序列中所有其他元素 $a^i$ 的关注程度，我们用 $a^1$ 的查询向量 $q^1$ 去和**所有**元素的键向量 $k^i$ 进行匹配。
*   **计算方式**:
    *   **点积 (Dot-Product)**: 最常用、最高效的方式。`score = q · k`。Transformer使用的就是这种方式的变体（缩放点积注意力）。
    *   **加性 (Additive)**: `score = W * tanh(Wq * q + Wk * k)`。在Query和Key维度不同时可以使用。

*   **计算过程**:
    $$ \alpha_{1,i} = q^1 \cdot k^i $$
    我们计算出 $q^1$ 对 $k^1, k^2, k^3, \dots$ 的所有注意力分数 $\alpha_{1,1}, \alpha_{1,2}, \alpha_{1,3}, \dots$。这些原始分数代表了“匹配度”。

*   **归一化**: 将这些原始分数通过一个 **Softmax** 函数进行归一化，得到最终的注意力权重 $\hat{\alpha}_{1,i}$。Softmax使得所有权重的和为1，可以被理解为一种概率分布。
    $$ \hat{\alpha}_{1,i} = \frac{\exp(\alpha_{1,i})}{\sum_j \exp(\alpha_{1,j})} $$

**Step 3: 加权聚合 Value，得到输出**

有了注意力权重后，我们用这些权重去加权求和**所有**元素的**值向量 $v^i$**，得到最终的输出向量 $b^1$。
$$ b^1 = \sum_i \hat{\alpha}_{1,i} v^i $$
*   **直观理解**: 如果模型计算出 $a^1$ 应该高度关注 $a^5$（即 $\hat{\alpha}_{1,5}$ 很大），那么 $a^5$ 的信息（$v^5$）将在构成 $b^1$ 时占据主导地位。

**关键点**:
*   **并行计算**: 序列中所有元素的输出 $b^1, b^2, b^3, \dots$ 是**可以被同时计算出来**的，因为每个 $b^i$ 的计算都独立于其他的 $b^j$。这解决了RNN的并行化难题。
*   **全局依赖**: 计算任何一个 $b^i$ 时，都利用了**整个序列**的信息，因此它天生就能捕捉长距离依赖关系。
*   **参数学习**: $W^q, W^k, W^v$ 这三个矩阵是模型的核心参数，它们是在训练过程中通过梯度下降**学习**得到的。模型会学会如何生成最有利于当前任务的Q, K, V。

##### **多头注意力机制 (Multi-Head Attention)**

**动机**: 只用一组Q, K, V，相当于只从一个“角度”或“子空间”去理解序列中元素间的关系。但元素间的关系是多样的（比如，既有语法关系，又有语义关系）。

**做法**:
1.  设置多个“头”（`n_heads`）。每个头都有**自己独立**的一套 $W_j^q, W_j^k, W_j^v$ 权重矩阵。
2.  对于每个头 $j$，都独立地执行一遍完整的Self-Attention计算，得到一个输出向量 $b^{i,j}$。
3.  将所有头产生的输出向量 $b^{i,1}, b^{i,2}, \dots$ **拼接 (Concatenate)** 起来。
4.  将拼接后的长向量通过一个额外的、可学习的线性层（权重为 $W^O$）进行融合和降维，得到最终的多头注意力输出 $b^i$。
    $$ b^i = W^O \cdot \text{concat}(b^{i,1}, b^{i,2}, \dots) $$

**好处**: 允许模型在不同的表示子空间中，同时关注来自不同位置的信息，从而捕捉到更丰富、更多样的特征和关系。

##### **位置编码 (Positional Encoding)**

**问题**: Self-Attention机制本身是**置换不变 (Permutation-invariant)** 的。也就是说，它不包含任何关于元素在序列中**位置或顺序**的信息。对于“A打了B”和“B打了A”这两个句子，如果不考虑位置，它们的词向量集合是相同的，模型将无法区分。

**解决方案**:
1.  为序列中的每一个位置 $i$ 创建一个**独特的位置编码向量 $e^i$**。
2.  将这个位置编码向量**直接加到**对应位置的输入词嵌入向量上。
    $$ \text{input}_i = a^i + e^i $$
3.  **编码方式**:
    *   **手工设计 (Hand-crafted)**: Transformer论文中使用的是基于 **sin 和 cos 函数**的周期性函数。这种方法的好处是模型可以泛化到比训练时遇到的更长的序列。
    *   **学习得到 (Learned)**: 也可以将位置编码作为模型的可学习参数，让模型在训练过程中自己学会每个位置的最佳表示。

#### **第四部分：Self-Attention 的应用与变体**

##### **应用于语音 (Speech)**
*   **挑战**: 语音序列通常非常长（数千甚至数万帧），Self-Attention的计算复杂度是序列长度的平方 ($L^2$)，这在计算上是不可接受的。
*   **解决方案**: **截断自注意力 (Truncated Self-attention)**。在计算注意力时，不考虑整个序列，而是只考虑一个固定大小的窗口内的上下文。

##### **应用于图像 (Image)**
*   **视角**: 一张图片可以被看作是一个像素网格，或者是一系列被展平的图像块 (Patches) 的向量集合（如Vision Transformer, ViT的做法）。
*   **CNN 与 Self-Attention 的关系**:
    *   **CNN是Self-Attention的受限版本**: 一个CNN的卷积核可以被看作是一种特殊的、权重固定的Self-Attention，它只能关注一个**固定的、局部的感受野**内的信息。
    *   **Self-Attention是CNN的泛化版本**: Self-Attention可以被看作是一种拥有**动态的、可学习的、全局感受野**的CNN。它在每次计算时，都会根据内容动态地决定“感受野”的形状和范围。

##### **应用于图 (Graph)**
*   在图神经网络（GNN）中，标准的Self-Attention可以让每个节点关注图中所有其他节点。
*   **考虑边的信息**: 可以修改Attention机制，使其**只在有边直接相连的节点之间**计算注意力分数。这引入了图的结构偏置，使得信息沿着图的边进行传播，这正是图注意力网络（GAT）的核心思想。

#### **第五部分：Self-Attention 与 RNN 的对比**

| 特性 | RNN (如LSTM, GRU) | Self-Attention (Transformer) |
| :--- | :--- | :--- |
| **计算方式** | 顺序计算，t时刻的计算依赖于t-1时刻的结果 | 并行计算，所有位置的输出可以同时计算 |
| **长距离依赖** | 理论上可以，实践中因梯度消失/爆炸而**困难** | **天生擅长**，通过直接的全局连接捕捉 |
| **路径长度** | 任意两个位置间的信息传递路径长度是 $O(L)$ | 任意两个位置间的信息传递路径长度是 $O(1)$ |
| **计算复杂度** | $O(L \cdot d^2)$ (L为长度, d为维度) | $O(L^2 \cdot d)$，**在长序列上成本高** |
| **位置信息** | 结构本身隐式包含 | 需要额外的位置编码来引入 |

**总结**: Self-Attention以其强大的并行计算能力和优异的长距离依赖捕捉能力，已经成为现代深度学习（尤其是在NLP领域）处理序列和集合数据的基石，并在向计算机视觉、语音等领域快速渗透。

当然，这是一个非常核心且重要的问题。Attention机制之所以具有革命性，正是因为它优雅地解决了输入向量长度可变的问题。

它主要通过以下**三个环环相扣的设计**来实现的：

---

### **1. 权重共享的参数化 (Shared Parameterization)**

首先，我们回顾一下传统模型（如全连接网络）为什么处理不了变长输入：
*   一个标准的全连接层，其输入层神经元的数量是**固定**的。如果输入一个长度为10的向量序列，你需要一个能接收10个向量的输入层；如果输入长度为20的序列，你就需要一个完全不同的、能接收20个向量的输入层。网络的结构直接被输入长度“写死”了。

**Attention机制的解决方案**：
它不为每个输入位置都创建一套独立的参数。相反，它只学习**三套固定的、与位置无关的权重矩阵**：$W^q$, $W^k$, 和 $W^v$。

*   **无论输入序列有多长**（是5个词还是50个词），模型都是用**同一套** $W^q, W^k, W^v$ 去处理序列中的**每一个**向量。
*   对于序列中的任意一个输入向量 $a^i$，它都会被这三套共享的权重矩阵转换为对应的 $q^i, k^i, v^i$。
*   这意味着，模型的参数数量与输入序列的长度**完全解耦**。模型的“骨架”是固定的，可以灵活地“套”在任意长度的序列上。

**这解决了“模型结构依赖于输入长度”的根本问题。**

---

### **2. 动态的、基于内容的交互矩阵 (Dynamic, Content-based Interaction)**

现在模型可以处理任意长度的输入了，但如何让不同长度的序列内部进行有意义的交互呢？

**Attention机制的解决方案**：
它通过计算一个**动态的“注意力分数矩阵”**来解决这个问题。这个矩阵的维度就是 `(序列长度, 序列长度)`。

1.  **查询与匹配 (Query-Key Interaction)**: 对于序列中的每一个元素（我们称之为“查询”`Query`），它都会去和序列中的**所有**元素（包括它自己，我们称之为“键”`Key`）进行一次“匹配度”计算。
    *   如果输入序列长度为 L，那么每个 `Query` 都会得到 L 个注意力分数。
    *   总共会进行 L x L 次匹配计算，形成一个 L x L 的注意力分数矩阵。

2.  **内容决定权重 (Content Determines the Weights)**: 最关键的是，这个 L x L 矩阵中的值**不是预设的，而是完全由输入内容动态计算出来的**。语义上更相关的词对，其对应的注意力分数就更高。

**这解决了“如何在不同长度的序列内部建立联系”的问题。** 无论序列多长，Attention总能通过一个动态生成的、尺寸为 L x L 的注意力矩阵来捕捉其内部所有元素之间的关系。

---

### **3. 归一化和加权求和 (Normalization and Weighted Sum)**

最后一步，是如何将这些动态计算出的关系（注意力权重）应用起来，并生成一个固定维度的输出。

**Attention机制的解决方案**：
1.  **Softmax 归一化**: 对每一个“查询”元素得到的那一行注意力分数（长度为L）进行Softmax操作。这使得该行所有分数的和为1，可以被看作是注意力的“概率分布”。无论L是5还是50，Softmax都能将其处理成一个有效的概率分布。

2.  **加权求和**: 将这个概率分布（L个权重）去乘以序列中**所有**元素对应的“值”向量（L个 `Value` 向量），然后将它们**相加**。
    $$ b^i = \sum_{j=1}^{L} \hat{\alpha}_{i,j} v^j $$
    无论L是多少，这个加权求和操作的最终结果都是一个**单一的、维度与`Value`向量相同的输出向量** $b^i$。

**这解决了“如何从可变数量的内部关系中，提炼出一个固定尺寸的、信息丰富的表示”的问题。**

---

### **总结与比喻**

你可以把这个过程想象成**开一场动态的圆桌会议**：

*   **输入序列可变**: 意味着参会人数（向量数量）可以是5个人，也可以是50个人。
*   **权重共享 (Wq, Wk, Wv)**: 意味着会议规则是固定的，不管多少人参加，每个人发言前都要先想好自己的“议题(Query)”、“身份牌(Key)”和“详细观点(Value)”。
*   **动态交互 (Attention Matrix)**: 会议开始后，每个想发言的人（Query），都要先听一圈所有人的“身份牌”（Keys），根据相关性决定自己等下要重点听谁的发言。参会人数越多，这个“关系表”就越大。
*   **加权求和**: 每个人最终形成的观点（输出向量 $b^i$），是综合了会场上所有人的“详细观点”（Values）之后的结果，并且重点参考了他认为最重要的那些人的观点。不管有多少人发表了观点，他最终都只形成**一个**综合性的观点。

通过这三步，Attention机制不仅能够自如地处理任意长度的输入序列，还能在没有循环结构的情况下，一步到位地捕捉序列中任意两个元素之间的长距离依赖关系，展现了其强大的灵活性和建模能力。