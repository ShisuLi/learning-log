# CNN

### A neuron does not have to see the whole image
- Some patterns are much smaller than the whole image
- Receptive field: all channels, kernel size 3*3,  stride 1, overlap, padding
- Each receptive field has a set of neurons (e.g., 64)
- The receptive fields cover the whole image

### The same patterns appear in different regions
- Two neurons with the same receptive field would not share parameters
- Share weights across different regions, filters

Fully Connected Layer - Receptive Field - Parameter Sharing - Convolutional Layer

- MLP: Jack of all trades, master of none
- CNN: Larger model bias, for image

**Neuron Version Story**
- Each neuron only considers a receptive field
- The neurons with different receptive fields share the same weights

**Filter Version Story**
- There are a set of filters detecting small patterns
- Each filter convolves over the input image

### Subsampling the pixels will not change the object

- pooling is a operator
- Max pooling, not necessary if computation cost is not a concern

Image - Convolution - Pooling - Convolution - Pooling - Flatten - Fully Connected Layer - Softmax - Output

AlphaGo

Network - Next move(19*19 positions, black 1, white -1, none 0) 19*19 classes
Fully connected network can be used, but CNN performs much better
19*19 matrix, 48 channels in AlphaGo

why cnn?
- Some patterns are much smaller than the whole image
- The same patterns appear in different regions
- AlphaGo does not use Pooling

CNN is not invariant to rotation and scaling, we need data augmentation


好的，我们来将这份关于卷积神经网络（CNN）的笔记，结合其核心原理与实际应用，扩展成一份全面、详细且易于理解的中文深度学习材料。

---

### **卷积神经网络 (Convolutional Neural Network, CNN) - 计算机视觉的基石**

传统的全连接网络（MLP）存在一个致命问题：当处理图像时，参数量会发生爆炸。一张小小的100x100像素的彩色图片，其输入维度就是100 * 100 * 3 = 30000。如果第一个隐藏层有1000个神经元，仅这一层的权重数量就高达30000 * 1000 = 3千万，这使得模型极难训练且容易过拟合。

CNN的诞生正是为了解决这个问题。它并非一个全新的模型，而是对全连接网络的一种**“加了限制”的特殊版本**。这种限制基于对图像数据的深刻洞察，从而引入了强大的**模型偏置（Model Bias）**。

#### **CNN的核心洞察与三大特性**

CNN的强大能力源于它对图像处理的三大直觉性假设：

1.  **局部性 (Locality): 神经元无需看到全局**
    *   **洞察**: 在图像中，有意义的模式（如一只眼睛、一个鸟嘴、一个轮胎的边缘）通常只占图像的一小部分。我们不需要让一个神经元一次性处理整张图片来识别这些局部模式。
    *   **实现 - 感受野 (Receptive Field)**: CNN中的每个神经元只与输入图像的一个局部区域相连，这个区域就是该神经元的“感受野”。
        *   **细节**: 一个感受野通常很小（如3x3、5x5像素），但它会覆盖输入图像的**所有通道 (all channels)**。例如，对于一张RGB彩色图像，一个3x3的感受野实际的维度是 **3x3x3**。
        *   **工作方式**: 神经元们（通常是一整组，例如64个）共同观察这同一个感受野，每个神经元负责检测一种特定的微小模式（如一条竖线、一个红色的斑点等）。
        *   **覆盖全局**: 这些感受野会像一个滑动的窗口一样，按照设定的**步长 (stride)**，从左到右、从上到下地扫过整张输入图像，以确保所有区域都被观察到。**填充 (Padding)** 技术则常用于处理图像边缘，确保边缘信息不被丢失。

2.  **平移不变性 (Translation Invariance): 同样模式，不同位置**
    *   **洞察**: 一个特定的模式（比如“猫耳朵”）可能出现在图像的左上角，也可能出现在右下角。但无论出现在哪里，它都是“猫耳朵”。因此，用于检测“猫耳朵”的神经元（或方法）应该是相同的。
    *   **实现 - 参数共享 (Parameter Sharing / Weight Sharing)**: 这是CNN最核心、最天才的思想。它规定：**负责检测同一种模式的神经元，在扫描图像的不同位置时，使用完全相同的权重和偏置**。
        *   **对比**: 在全连接网络中，如果两个神经元的感受野不同（即连接到输入的不同位置），它们的权重也必然不同。
        *   **CNN的做法**: CNN中负责检测“竖线”的这组神经元，无论它的感受野滑动到图像的哪个位置，其内部的权重都是**共享**的。
        *   **巨大优势**: 参数共享极大地减少了模型的参数数量。我们不再需要为每个位置都学习一套独立的权重，而只需要学习一套能识别特定模式的权重即可。

3.  **层级性 (Hierarchy): 从简单到复杂的模式组合**
    *   **洞察**: 复杂的物体是由简单的模式逐级组合而成的。例如，边缘组合成纹理，纹理组合成物体的部件（如鼻子、眼睛），部件再组合成完整的物体（如一张脸）。
    *   **实现 - 深度堆叠 (Stacking Layers)**: CNN通过堆叠多个卷积层来实现这种层级化的特征提取。
        *   **浅层网络**: 靠近输入的卷积层，其感受野较小，学习到的是非常基础的模式，如颜色、边缘、角点等。
        *   **深层网络**: 靠近输出的卷积层，它所“看到”的输入不再是原始像素，而是前几层提取出的基础模式图。因此，深层网络能在此基础上学习到更复杂、更抽象的模式，如“眼睛”、“车轮”等。

#### **从全连接到卷积：一种思想的演进**

**全连接网络 (MLP) <-> 卷积网络 (CNN)**

1.  **全连接层 (Fully Connected Layer)**: 每个神经元都连接到前一层的所有输出。这就像一个知识渊博但没有专长的**“万事通” (Jack of all trades, master of none)**。它有能力学习任何可能的模式，但因为它没有针对图像的任何先验知识（偏置），所以需要海量的数据才能学好，且参数冗余。

2.  **引入“感受野”**: 我们限制每个神经元只看一个局部区域。这相当于告诉模型：“别看了，全局信息没用，先从局部细节学起。” 这大大减少了连接数。

3.  **引入“参数共享”**: 我们强迫检测同一模式的神经元在不同位置使用相同的权重。这相当于告诉模型：“这个‘竖线’检测器在左上角好用，在右下角也应该好用，别重复学习了。” 这极大地减少了参数量。

经过这两步限制，一个臃肿的全连接层就演变成了轻便、高效的**卷积层 (Convolutional Layer)**。CNN因此拥有了更强的**模型偏置 (Model Bias)**，它天生就适合处理具有局部性和平移不变性的图像数据。

#### **CNN的两种叙事角度 (双重理解)**

理解卷积层的运作方式，可以从两个等价但视角不同的“故事”出发：

*   **故事一：从神经元的角度 (Neuron Version)**
    *   网络中有一群“专家”神经元。
    *   每个“专家”只负责一个非常小的**感受野**，专注于检测一种微小的局部模式。
    *   不同“专家组”负责检测不同的模式（比如一组检测横线，另一组检测斜线）。
    * 整个图像上不同位置各有一个专家组，不同专家组中检测相同模式的神经元**共享权重**。
    *   所有“专家”在所有位置的判断结果，共同构成一张新的特征图 (Feature Map)。

*   **故事二：从滤波器的角度 (Filter Version)**
    *   我们定义了一组**滤波器 (Filter) 或卷积核 (Kernel)**。每个滤波器本质上就是一个小尺寸的权重矩阵，它被设计用来检测一种特定的视觉模式。例如，一个用于检测垂直边缘的滤波器可能长这样：`[[1, 0, -1], [1, 0, -1], [1, 0, -1]]`。
    *   每个滤波器在整个输入图像（或前一层的特征图）上进行**卷积 (Convolve)** 运算。卷积的数学过程，就是滤波器在图像上滑动，并将滤波器自身的权重与图像对应位置的像素值进行加权求和。
    *   这个加权求和的结果，如果值很大，就说明该位置与滤波器所要检测的模式非常匹配。
    *   每个滤波器经过卷积运算后，都会生成一张新的二维**特征图**，这张图代表了该滤波器在原始图像各个位置的激活强度。
    *   如果有64个滤波器，我们就会得到64张特征图，它们堆叠在一起，成为下一层的输入。

这两个故事描述的是同一件事：**局部检测 + 权重共享**。

#### **池化层 (Pooling) - 降维与不变性**

**洞察**: 对于识别一个物体来说，图像的细节（像素）在一定程度上被**二次采样 (Subsampling) 或轻微扭曲，并不会改变物体的本质**。一只猫稍微放大或缩小，它仍然是猫。

*   **作用**: 池化层的主要目的是对特征图进行**降维 (Down-sampling)**。它通常跟在卷积层之后，通过一种简单的运算来减小特征图的空间尺寸（宽度和高度）。
*   **好处**:
    1.  **降低计算成本**: 特征图变小，后续卷积层的参数量和计算量也随之减少。
    2.  **增加感受野**: 池化使得后续卷积层的神经元能够看到更广阔的原始输入区域。
    3.  **提供小范围的平移不变性**: 例如，最大池化（Max Pooling）会取一个区域内的最大值。只要那个最强的特征信号还留在这个区域内，无论它在这个区域内如何轻微移动，池化后的结果都可能保持不变。
*   **操作 (Operator)**: 池化是一种预设的、**没有可学习参数**的操作。最常见的是**最大池化 (Max Pooling)**，它将特征图划分为若干个小区域，并只取每个区域内的最大值作为输出。平均池化（Average Pooling）则是取平均值。
*   **非必需性**: 如果计算成本不是主要问题，池化层并非不可或缺。一些现代网络（如ResNet的部分变体）会使用步长为2的卷积层来替代池化层，同样可以实现降维的效果。

#### **典型的CNN架构：串联与整合**

一个经典的CNN架构就像一条处理图像的流水线：

**输入图像 -> [卷积层 -> 激活函数 -> 池化层]*N -> 展平层 -> [全连接层]*M -> Softmax -> 输出**

1.  **[卷积层 -> ... -> 池化层]** 模块被重复N次。这个部分是CNN的**特征提取器 (Feature Extractor)**。随着网络的加深，提取的特征越来越抽象和高级。
2.  **展平层 (Flatten Layer)**: 在经过多轮卷积和池化后，得到的高级特征图是多维的。展平层的作用就是把它“拉直”成一个一维的长向量。
3.  **[全连接层]** 模块被重复M次。这个部分扮演着**分类器 (Classifier)** 的角色。它接收展平后的高级特征向量，并像一个传统的MLP一样，对其进行分析和整合，最终做出分类判断。
4.  **Softmax层**: 输出最终的分类概率。

#### **CNN的应用延申：以AlphaGo为例**

CNN的威力远不止于识别猫和狗。AlphaGo的成功就极大地借鉴了CNN的思想。

*   **问题建模**: 围棋棋盘是一个19x19的网格。AlphaGo的其中一个网络（策略网络）的任务是，给定当前棋盘的局面，预测下一步最有价值的落子位置。这是一个19x19=361类的分类问题。
*   **为什么不用全连接网络？**: 当然可以用，但效果会差很多。因为围棋的决策同样高度依赖于CNN的核心洞察：
    1.  **局部性**: 一个棋子的“气”、一个棋形的“眼”，这些都是由周围少数几个棋子决定的**局部模式**。
    2.  **平移不变性**: 一个经典的“虎口”棋形，无论出现在棋盘的哪个角落，其战术意义都是相似的。
*   **AlphaGo的做法**:
    *   它将19x19的棋盘看作一张19x19的**图像**。
    *   棋盘上的状态（黑子、白子、空位）以及其他信息（如气、历史落子记录等）被编码成不同的**通道 (Channel)**。AlphaGo的输入有多达48个通道！
    *   然后，它使用一个深度CNN来分析这个19x19x48的“图像”，提取棋局中的各种战术模式，最终输出一个19x19的概率分布图，代表在每个位置落子的推荐概率。
    *   **AlphaGo没有使用池化层**，因为它需要精确地知道每个位置的信息，降维会丢失关键的位置细节。

#### **CNN的局限性与展望**

*   **旋转与缩放不变性的缺失**: 传统的CNN对于**旋转 (Rotation)** 和 **缩放 (Scaling)** 并不具备内在的不变性。一张正着的人脸和一张倒着的人脸，在CNN看来是完全不同的。
*   **解决方案**: **数据增强 (Data Augmentation)**。在训练时，通过对原始图像进行随机的旋转、缩放、裁剪、翻转等操作，人为地创造出更多样化的训练样本。这迫使模型去学习到，这些变换后的图像本质上还是同一个物体。
*   **未来**: 更先进的架构，如**Transformer**在视觉领域的应用（Vision Transformer, ViT），正在挑战CNN在某些任务上的统治地位。但CNN凭借其高效、成熟和强大的模型偏置，在可预见的未来仍将是计算机视觉领域的核心技术。