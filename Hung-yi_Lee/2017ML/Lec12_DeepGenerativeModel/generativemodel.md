# Deep Generative Model
## PixelRNN
- To create an image, generating a pixel each time.
- Model the conditional distribution of each pixel given the previously generated pixels.
- WaveNet

## VAE
input-encoder-m1,m2,m3 sigma1,2,3, e1,2,3(from a normal distribution)-ci=exp(sigmai)*ei+mi-decoder-output
- 引入了噪音，code space
- Minimize $\sum^3_{i=1}(exp(\sigma_i)-(1+\sigma_i)+(m_i)^2)$, L2 regularization, we want sigmai close to 0, variance close to 1

### Gaussian Mixture Model
- Each x you generate is from a mixture of several Gaussian distributions.
- Distributed representation is better
- m~P(m) (multinomial), x|m ~ N(mu^m, sigma_m)
- P(x) = sum_m P(m)P(x|m)

- z is a vector from normal distribution
- Even though z 
- Each dimension of z represents an attribute
- z~N(0,I), x|z~N(mu(z), sigma(z))
- Decoder: 训练NN，输入z，输出mu(z), sigma(z)
- $P(x) = \int_z P(x|z)P(z) dz$
- $L=\sum_x \log P(x)$ maximizing the likelihood of the observed x
- Encoder: 训练NN，输入x，输出mu'(x), sigma'(x)
- $q(z|x) = N(\mu'(x), \sigma'(x))$

- it does not really try to simulate real images; VAE may just memorize the existing images, instead of generating new images.


## GAN
- Discriminator: input image, output real/fake
- Generator: input random noise, output image
randomly sample a vector -- NN generator -- Discriminator -- real/fake
- fix the discriminator, use gradient descent to update the generator
- 不确定discriminator是否更强 No explicit signal about how good the generator is.
- We have to keep well-matched in a contest. 
- When discriminator fails, it does not guarantee that generator generates realistic images.
    - Just because discriminator is stupid
    - Sometimes generator find a specific example that can fail the discriminator
- GANs are difficult to train, and often suffer from mode collapse, a phenomenon where the model learns to generate only a limited variety of outputs.


好的，我们来将这份关于深度生成模型（Deep Generative Models）的笔记进行一次全面、深入、且结构化的翻译与扩展讲解。这份文档将重点剖析PixelRNN、VAE和GAN这三种里程碑式的生成模型，澄清它们的核心思想、数学原理、训练技巧以及各自的优缺点。

---

### **深度生成模型（Deep Generative Models）：代码与思想的深层解读**

这份笔记聚焦于三种经典的深度生成模型，它们从不同角度解决了“如何让机器创造”这一核心问题。

#### **第一部分：PixelRNN - 像写文章一样画画**

##### **核心思想 (Basic Idea)**

*   **PixelRNN** 是一种**自回归（Autoregressive）**模型。它将生成图像的过程类比为生成文本的过程：**一次只生成一个像素（a pixel each time）**。
*   **条件分布建模**: 它的核心是**对每个像素的条件概率分布进行建模（Model the conditional distribution）**。具体来说，要生成第 `i` 个像素的颜色，模型需要依赖于它之前已经生成的所有 `1` 到 `i-1` 个像素的颜色。生成顺序通常是逐行（row by row）或逐像素（pixel by pixel）。
    *   数学上，它对联合概率分布 $P(\text{image})$ 进行链式分解：$P(\text{image}) = \prod_i P(\text{pixel}_i | \text{pixel}_1, ..., \text{pixel}_{i-1})$。

##### **与WaveNet的联系**

*   **WaveNet** 是一个用于生成原始音频波形（raw audio waveform）的自回归模型，由DeepMind提出。它与PixelRNN在思想上是完全一致的，都是一次生成一个数据点（音频采样点），且每个点的生成都依赖于之前的所有点。
*   WaveNet通过**因果空洞卷积（Causal Dilated Convolutions）**极大地扩展了感受野，使得模型在生成当前采样点时，能高效地回顾过去很长一段时间的音频信息，这对于捕捉音频的长程依赖至关重要。PixelRNN也借鉴了类似的思想来处理二维图像的依赖关系。

##### **优缺点**

*   **优点**:
    *   **精确的似然估计**: 可以直接、精确地计算出任一给定图像的似然概率 $P(\text{image})$，这使得它在概率建模上非常强大。
    *   **高质量的局部结构**: 由于是逐像素生成，它能很好地捕捉像素间的局部相关性。
*   **缺点**:
    *   **极度缓慢的生成速度**: 生成一张图片需要成千上万次的顺序前向传播，计算成本极高，实用性差。

---

#### **第二部分：变分自编码器 (VAE) - 在“噪声”中学习平滑的创意空间**

##### **结构与随机性的引入**

*   **VAE** 不是一个确定性的自编码器，它在**隐空间（code space）**中巧妙地**引入了随机性（噪声）**。
*   **工作流程**:
    1.  **编码 (Encoder)**: 输入图像 `x`，编码器输出的不是一个单一的隐向量，而是该图像在隐空间中对应的**高斯分布的参数**：一组均值向量 `m = (m1, m2, m3, ...)` 和一组对数方差向量 `log(σ^2) = (σ1, σ2, σ3, ...)`。（笔记中写 `sigma` 可能指对数标准差或对数方差，但概念一致）。
    2.  **采样 (Sampling)**: 从一个标准正态分布 `N(0, I)` 中采样一个噪声向量 `e = (e1, e2, e3, ...)`。
    3.  **重参数化技巧**: 使用这些参数和噪声，计算最终的隐向量 `c`：`ci = exp(σi/2) * ei + mi`。（笔记中的 `exp(sigma_i)` 是 `exp(log(variance)/2)` 即标准差）。这一步至关重要，它将随机的采样过程与可微分的计算过程解耦。
    4.  **解码 (Decoder)**: 将隐向量 `c` 输入解码器，重构出输出图像。

##### **独特的损失函数**

VAE的训练目标是最大化观测数据的对数似然的**证据下界（Evidence Lower Bound, ELBO）**，这等价于最小化一个由两部分组成的损失函数：

1.  **重构损失**: 与标准自编码器一样，衡量输入图像与输出图像之间的差异（如MSE）。
2.  **KL散度损失**: 这是VAE的精髓。它衡量编码器输出的分布 `q(z|x) = N(m, σ^2)` 与一个固定的**先验分布**（通常是标准正态分布 `N(0, I)`）之间的差异。
    *   笔记中给出的公式 `sum(exp(σi) - (1 + σi) + mi^2)` 正是 `N(m, exp(σ))` 与 `N(0, I)` 之间KL散度的解析形式。
    *   这个损失项就像一个**L2正则化项**，它强迫编码器产生的隐空间分布向标准正态分布“靠拢”。这使得整个隐空间变得规整、连续、没有“空洞”，并且以原点为中心。
    *   `we want sigmai close to 0, variance close to 1`：当对数方差 `σi` 接近0时，方差 `exp(σi)` 就接近1。这正是标准正态分布的方差。

##### **VAE与高斯混合模型 (GMM) 的类比**

*   **高斯混合模型 (GMM)**: 认为每个数据点 `x` 是从多个高斯分布的混合中生成的。`P(x) = sum_m P(m)P(x|m)`，其中 `m` 是一个离散的类别变量。
*   **VAE的连续扩展**: VAE可以被看作是GMM的一个**无限、连续版本**。
    *   不再有有限个离散的类别 `m`，而是有一个连续的、服从标准正态分布的隐向量 `z`。`z ~ N(0, I)`。
    *   `P(x) = ∫ P(x|z)P(z) dz`，通过积分，将所有可能的 `z` “混合”起来生成 `x`。
    *   **分布式表示 (Distributed representation is better)**: 相比于GMM中每个数据点只属于一个高斯分量，VAE中的 `z` 是一个高维向量。它的**每个维度都可以独立地代表数据的一个潜在属性（attribute）**，这是一种更强大、更高效的表示方式。

##### **对VAE的批判性思考**

*   **它真的在“创造”吗？**: `it does not really try to simulate real images; VAE may just memorize the existing images, instead of generating new images.`
    *   这是一种对早期或训练不佳的VAE的批评。由于VAE的损失函数中包含强大的重构项，如果KL散度损失的权重过低，模型可能会退化为一个普通的自编码器，倾向于“记住”训练数据，而不是学习一个泛化的、能创造新样本的生成模型。
    *   同时，由于MSE等重构损失的“平均化”效应，VAE生成的图像往往比较**模糊**，缺乏真实感。

---

#### **第三部分：生成对抗网络 (GAN) - 伪造者与鉴定师的博弈**

##### **核心组件与博弈过程**

*   **生成器 (Generator)**:
    *   **输入**: 一个从简单分布（如正态分布）中随机采样的噪声向量。
    *   **输出**: 一张伪造的图像。
    *   **目标**: 生成尽可能逼真的图像，以“骗过”判别器。
*   **判别器 (Discriminator)**:
    *   **输入**: 一张图像（可能是真实的，也可能是伪造的）。
    *   **输出**: 一个概率值，代表该图像是“真实”的概率。
    *   **目标**: 尽可能准确地分辨出真实图像和伪造图像。

*   **训练循环**:
    1.  **固定生成器，更新判别器**: 生成一批假图，与一批真图混合，训练判别器进行分类。
    2.  **固定判别器，更新生成器**: `fix the discriminator, use gradient descent to update the generator`。生成器生成一张假图，并将其喂给判别器得到一个“真实度”分数。生成器会**以最大化这个分数为目标**，通过梯度下降来更新自己的参数，学习如何生成能得高分的图片。

##### **训练的挑战与不确定性**

*   **没有明确的收敛信号**: `No explicit signal about how good the generator is.` 我们没有像VAE的重构误差那样的直接指标来衡量生成器的“好坏”。生成器的损失完全来自于判别器的反馈，这个反馈本身是动态变化的。
*   **匹配竞赛**: `We have to keep well-matched in a contest.` 生成器和判别器的能力需要保持动态平衡。如果一方过于强大，另一方就无法从中学到有效的信息，导致训练崩溃。
*   **判别器失效的误导**: `When discriminator fails, it does not guarantee that generator generates realistic images.`
    *   **原因1: 判别器太笨**: `Just because discriminator is stupid`。判别器可能因为自身能力不足而被轻易骗过，但这并不代表生成器生成的图像真的很好。
    *   **原因2: 对抗样本**: `Sometimes generator find a specific example that can fail the discriminator`。生成器可能非常“狡猾”，它不去学习整个真实图像的分布，而是找到了一个或几个特定的“对抗样本”，这些样本虽然看起来很奇怪，但恰好能戳中当前判别器的“盲点”，让其做出错误的判断。
*   **模式崩溃 (Mode Collapse)**: `GANs are difficult to train, and often suffer from mode collapse.` 这是GAN最著名的问题之一。生成器可能会发现，只生成一种或少数几种特别逼真的图像（例如，只生成同一种类型的狗），就能稳定地骗过判别器。于是它就放弃了探索数据的多样性，导致生成结果高度单一。