# Autoencoder
## Basic idea of autoencoder
Autoencoder
Denoising Autoencoder

BERT is a de-noising auto-encoder:
Masked--Add noise
BERT--Encoder
logits--Embedding
Linear+softmax--Decoder
Minimize cross entropy--Reconstruction

## Feature disentanglement
- Representation includes information of different aspects
- 得知embedding各个维度分别代表什么
- 可以做voice conversion，提取表示内容的部分和表示语音特征的部分，拼接

## Discrete Latent Representation
- 强迫Real numbers变成离散值 Binary、one-hot
- 在没有 label的情况下让模型学会分类

### VQVAE
- Vector Quantized Variational Autoencoder
- Codebook: a set of embedding vectors, learn from data
- Compute similarity, the most similar one is the input of decoder
input -- nn encoder -- codebook -- vector n -- nn decoder -- output

### Text as Representation
document -- Encoder Seq2seq -- word sequence, summary? -- Decoder Seq2seq -- document
- seq2seq2seq auto-encoder
- only need a lot of documents to train the model, unsupervised summarization
- but the word sequence may be unreadable
- Discriminator, learned human written summaries, discriminate the summary real or not

## More Applications
- Decoder as a generator: randomly generate a vector from a distribution, input to decoder to generate image. With some modification, we have variational auto-encoder (VAE)
- Compression, Lossy
- Anomaly Detection: given a set of training data, detecting input x is similar to training data or not.
    - Binary classification, but only have one class data
    - Training: Using real human faces to learn an auto-encoder
    - Testing: input a face image, compute reconstruction error
        - If the error is small, it is similar to training data
        - If the error is large, it is different from training data, may be an anomaly

好的，我们来将这份关于自编码器（Autoencoder）的笔记进行一次全面、深入且结构化的翻译与扩展讲解。这份文档将从自编码器的基本思想出发，逐步深入到其高级变体、核心概念（如特征解耦）、以及在各个领域的广泛应用。

---

### **自编码器 (Autoencoder)：数据自身的无监督压缩与重构**

自编码器（Auto-encoder）是一类特殊的、以**无监督**方式训练的神经网络。其核心目标非常直观：**学习一个输入数据的紧凑表示（压缩），然后用这个紧凑表示来尽可能完美地重构出原始输入（解压）**。

这个过程就像是你写了一本书，然后让一个助手帮你写一份书的摘要（编码），再让另一个助手仅根据这份摘要把你的原书重新写出来（解码）。如果最终重写出来的书和你的原作一模一样，那就说明这份“摘要”抓住了书的精髓。

#### **第一部分：自编码器的基本思想**

##### **1. 标准自编码器 (Basic Autoencoder)**

*   **结构**:
    *   **编码器 (Encoder)**: 接收原始的高维输入数据（如一张图片），并将其压缩成一个低维的、信息密集的**隐向量 (Latent Vector)** 或**编码 (Code)**。这个过程也叫**表示学习 (Representation Learning)**。
    *   **解码器 (Decoder)**: 接收编码器输出的隐向量，并尽力将其还原（重构）成与原始输入一模一样的数据。
*   **训练目标**: 最小化**重构误差 (Reconstruction Error)**，即原始输入与解码器输出之间的差异（例如，对于图像，可以使用均方误差 MSE）。
*   **瓶颈 (Bottleneck)**: 编码器输出的隐向量维度通常远小于输入维度，这个“中间层”被称为模型的“瓶颈”。正是这个瓶颈，强迫模型不能简单地“复制粘贴”输入，而是必须学习到数据中**最重要、最本质的特征**，才能在解码时成功还原。

##### **2. 去噪自编码器 (Denoising Autoencoder)**

这是对基本自编码器的一个强大改进，使其表示能力更加鲁棒。

*   **核心思想**: 如果一个模型能从一个**被损坏（加入噪声）**的版本中恢复出**原始的、干净**的版本，那么它一定对数据的内在结构和模式有了更深刻的理解。
*   **工作流程**:
    1.  **加噪 (Add Noise)**: 在将原始输入喂给编码器之前，先人为地对其进行一些“破坏”，例如加入高斯噪声、随机遮盖掉一部分（Masking）。
    2.  **编码与解码**: 编码器接收这个损坏的输入，并尝试生成一个隐向量。解码器根据这个隐向量，进行重构。
    3.  **训练目标**: 最小化解码器的输出与**原始的、干净的输入**之间的重构误差。

##### **3. BERT：一个语言领域的去噪自编码器**

这个类比非常精妙，它揭示了BERT与自编码器思想的深层联系。

*   **加噪 (Add Noise)**: BERT的**Masked Language Model (MLM)** 任务，随机地将句子中的某些词元替换为`[MASK]`标记，这本质上就是一种对序列数据的“加噪”或“破坏”。
*   **编码器 (Encoder)**: BERT模型本身（即多层Transformer Encoder）负责处理这个被破坏的句子，并为每个词元生成一个富含上下文信息的表示（Embedding）。
*   **解码器 (Decoder)**: 在BERT的输出端，通常会接一个简单的**线性层 + Softmax**。它的任务是根据`[MASK]`位置的最终表示，预测出原始的词元应该是什么。这个预测过程，就可以看作是一个“解码”行为。
*   **重构损失 (Reconstruction Loss)**: BERT训练时优化的**交叉熵损失 (Cross Entropy Loss)**，其目标就是让模型预测的词元与原始的、未被Mask的词元尽可能一致。这与自编码器最小化重构误差的目标在精神上是完全一致的。

#### **第二部分：高级概念与变体**

##### **1. 特征解耦 (Feature Disentanglement)**

*   **目标**: 我们希望学习到的隐向量是“解耦”的，即其**每一个维度都独立地控制着数据中某一个有意义的、可解释的属性**。
*   **例子**: 对于人脸图像，一个理想的解耦表示可能是：
    *   维度1控制性别
    *   维度2控制年龄
    *   维度3控制头发颜色
    *   维度4控制是否戴眼镜
    *   ...
*   **应用：语音转换 (Voice Conversion)**: 这是一个绝佳的应用场景。
    1.  **解耦**: 训练一个自编码器，将一段语音的隐向量解耦成两个部分：**内容部分 (Content Representation)**，代表说了什么词；和**风格部分 (Style/Speaker Representation)**，代表是谁说的、用什么音色说的。
    2.  **转换**: 取源说话人A语音的“内容部分”，与目标说话人B语音的“风格部分”进行**拼接 (Concatenate)**，然后将这个新的组合向量输入解码器。
    3.  **结果**: 解码器将生成一段新的语音，其内容与A相同，但音色和风格与B完全一样，从而实现语音转换。

##### **2. 离散隐空间表示 (Discrete Latent Representation)**

*   **动机**: 传统的自编码器学习的是一个连续的（real numbers）隐空间。但在某些场景下，我们希望隐空间是**离散的**（例如，二进制向量或one-hot向量），这更符合分类、聚类等任务的本质。
*   **优势**: 强迫模型进行离散选择，可以在**没有任何标签**的情况下，让模型自动学会对数据进行**分类**。模型必须决定每个输入样本“属于”隐空间中的哪一个离散状态。

##### **3. VQ-VAE (Vector Quantized Variational Autoencoder)**

这是实现离散隐空间最成功、最著名的模型之一。

*   **核心组件：码本 (Codebook)**: VQ-VAE引入了一个共享的“码本”，它是一个包含多个（例如512个）embedding向量的集合。你可以把这个码本想象成一本调色板。
*   **工作流程**:
    1.  **编码**: 编码器将输入图片编码成一个中间向量。
    2.  **量化 (Quantization)**: 模型会去“查阅”码本，找到与编码器输出的中间向量**最相似**的那个码本向量（例如，调色板中最接近的颜色），并用这个码本向量作为最终的隐表示，输入给解码器。
    3.  **解码**: 解码器根据这个从码本中“查”到的离散向量，重构出图像。
*   **学习**: 在训练过程中，编码器、解码器和**码本本身**都会通过梯度下降进行学习，使得整个系统能找到最优的“调色板”和最佳的编解码方式。

##### **4. 文本作为表示 (Text as Representation)**

这是一个非常创新的想法，它将“摘要”这个比喻变成了现实。

*   **模型结构**: **Seq2Seq2Seq Autoencoder**
    1.  **编码器 (Encoder Seq2Seq)**: 输入一篇长文档，输出一个简短的、可读的**词序列**，例如自动生成的**摘要 (Summary)**。
    2.  **解码器 (Decoder Seq2Seq)**: 仅根据这个生成的摘要，尝试重构出原始的完整文档。
*   **优势**:
    *   **无监督摘要**: 只需要大量的文档数据即可训练，不需要人工撰写的“标准摘要”。
    *   **可解释性**: 它的隐空间表示不再是一个不可读的数字向量，而是一段人类可以理解的文本。
*   **挑战与改进**:
    *   **可读性问题**: 模型生成的中间摘要可能语法不通、难以理解。
    *   **引入判别器 (Discriminator)**: 可以额外引入一个判别器，它被训练用来区分“人类写的摘要”和“模型生成的摘要”。通过对抗训练，可以迫使编码器生成更自然、更像人话的摘要。

#### **第三部分：自编码器的更多应用**

##### **1. 作为生成器 (Generator)**

*   标准的自编码器不擅长生成，因为它的隐空间可能是不连续、有“空洞”的。
*   **变分自编码器 (VAE)**: 通过对其进行一些数学上的修改（如引入KL散度损失，强迫隐空间服从正态分布），我们可以确保隐空间是连续且结构良好的。这样，我们就可以**随机地从这个分布中采样一个向量，然后输入给解码器，从而生成全新的、逼真的图像**。

##### **2. 有损压缩 (Lossy Compression)**

*   自编码器的编码-解码过程，本质上就是一种**有损压缩**算法。编码器是压缩器，解码器是解压器。
*   与JPEG、MP3等传统算法不同，自编码器是一种**基于学习**的压缩方法。它可以针对特定类型的数据（如人脸、医学图像）进行优化，从而在极高的压缩率下，获得远超传统方法的视觉质量。

##### **3. 异常检测 (Anomaly Detection)**

这是自编码器最经典、最有效的应用之一。

*   **问题背景**: 在许多场景下，我们只有大量的“正常”数据，而“异常”数据非常稀少或根本没有。这使得传统的二分类方法难以奏效。
*   **基于自编码器的解决方案**:
    1.  **训练阶段**: **仅使用正常数据**（例如，大量真实的人脸图片）来训练一个自编码器。模型会学习到如何高效地压缩和重构“正常”的模式。
    2.  **测试阶段**: 当一个新数据到来时，将其输入训练好的自编码器，并计算其**重构误差**。
        *   **如果重构误差很小**: 说明这个新数据与训练时见过的正常数据非常相似，模型可以很好地处理它。它被判定为**正常**。
        *   **如果重构误差很大**: 说明这个新数据的模式是模型在训练时从未见过的，它不符合“正常”数据的内在结构，模型无法很好地重构它。它被判定为**异常**（Anomaly）。例如，输入一张卡通脸或者一只猫的图片，重构误差会非常大。