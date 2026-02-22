# Transformer
### Sequence to Sequence Model(Seq2Seq)
- Input a sequence, output a sequence.
- The output length is determined by model.
tasks 
- Speech Recognition, Machine Translation, Speech Translation
- Text-to-Speech (TTS) Synthesis
- Seq2seq for Chatbot, QA can be done by seq2seq
- Seq2seq for Syntactic Parsing文法剖析
- Seq2seq for Multi-label Classification: An object can belong to multiple classes
- Seq2seq for object detection

**Input sequence-Encoder-Decoder-output sequence**

x^i - Encoder - h^i
input embedding+positional encoding
blocks: self-attention, residual connection(input+output), layer norm, fc, residual connection, layer norm


#### Autoregressive 
BEGIN(special token)...+Encoder-Decoder-softmax-distribution, Vocabulary Size V(common characters, subwords, END)-max

#### Nonautoregressive Decoder
- BEGIN...BEGIN-NAT Decoder
- Another predictor for output length
- Output a very long sequence, ignore tokens after END
- Parallel, controllable output length
- 语音系统中常用，乘除调整classifier大小即可调整语音速度
NAT is usually worse than AT(multimodality problem)

#### Cross Attention(appear before self-attention)
- Decoder: q + Encoder: k -- alpha -- softmax -- + Encoder: v weighted sum -- FC
- minimize cross entropy
- Teacher Forcing: use ground truth as input during training

#### Copy mechanism
- Copy from input sequence to output sequence

#### Guided Attention
- In some tasks,  input and output are monotonically aligned. For example, speech recognition, TTS, etc.
- Monotonic Attention, Location-aware attention


### Beam Search
- Greedy Decoding: select the token with highest probability at each step
- Not possible to check all the paths
- Beam Search: keep the top-k paths at each step

- Another idea: Randomness is needed for decoder when generating sequence in some tasks(e.g., sentence completion, TTS)


### BLEU score
- Cross entropy & BLEU
- Cross entropy: training, token-level
- BLEU: evaluation, sentence-level, not differentiable; can use Reinforcement Learning to optimize BLEU
- BLEU: compare n-grams between generated sequence and reference sequence

### Scheduled Sampling


好的，我们来将这份关于Transformer的笔记，翻译、扩展并完善成一份全面、细致且反映最新知识的深度学习文档。这份文档将不仅解释笔记中的每个概念，还会补充其背后的原理、动机以及在实际应用中的考量。

---

### **Transformer：序列到序列模型的革命**

Transformer是Google在2017年的论文《Attention Is All You Need》中提出的一个革命性架构。它彻底抛弃了传统的循环（RNN）和卷积（CNN）结构，完全依赖于**注意力机制（Attention）**来处理序列数据，并成为了当今几乎所有大语言模型（如GPT、BERT）的基石。

#### **第一部分：序列到序列模型 (Sequence-to-Sequence, Seq2Seq)**

Seq2Seq模型是一类特殊的神经网络，其核心功能是：**输入一个序列，输出另一个序列，并且输出序列的长度由模型自己决定**。

##### **1. Seq2Seq模型的广泛应用**

这种灵活的输入输出模式使其能够解决众多看似不同但本质相似的任务：

*   **机器翻译 (Machine Translation)**: 输入一种语言的句子（序列），输出另一种语言的句子（序列）。
*   **语音识别 (Speech Recognition)**: 输入一段语音的声学特征序列，输出对应的文字序列。
*   **语音翻译 (Speech Translation)**: 输入语音序列，直接输出另一种语言的文字序列。
*   **文本到语音合成 (Text-to-Speech, TTS)**: 输入文字序列，输出可合成语音的声学特征序列（如梅尔频谱）。
*   **聊天机器人/问答 (Chatbot/QA)**: 输入一个问题或对话的上文（序列），输出一个回答（序列）。
*   **句法分析 (Syntactic Parsing)**: 输入一个句子，输出其对应的句法结构树（可以线性化为一个序列）。
*   **多标签分类 (Multi-label Classification)**: 将问题看作是“输入一张图片，输出其所有包含的类别标签序列”。
*   **目标检测 (Object Detection)**: 将问题看作是“输入一张图片，输出一系列边界框（Bounding Box）及其类别标签的序列”。

##### **2. Encoder-Decoder 经典架构**

几乎所有的Seq2Seq模型都遵循经典的**编码器-解码器 (Encoder-Decoder)** 架构。

*   **编码器 (Encoder)**: 负责“阅读”并“理解”整个输入序列。它将输入序列中的每个元素以及它们的上下文关系，压缩成一系列富含语义信息的中间表示（上下文向量）。
    *   **Transformer Encoder内部结构**:
        1.  **输入层**: 原始输入（如词向量）加上**位置编码（Positional Encoding）**，以引入序列的顺序信息。
        2.  **编码器块 (Encoder Block)**: 这个块会重复N次。
            *   **多头自注意力 (Multi-Head Self-Attention)**: 核心部件，让输入序列中的每个词都能“看到”并关联序列中的所有其他词，捕捉全局依赖关系。
            *   **残差连接 (Residual Connection) & 层归一化 (Layer Normalization)**: 将自注意力层的输入和输出相加，然后进行归一化。这有助于缓解梯度消失，稳定训练。
            *   **前馈网络 (Feed-Forward Network)**: 一个简单的全连接网络，对自注意力的输出进行非线性变换，增加模型的表达能力。
            *   **另一次残差连接 & 层归一化**。

*   **解码器 (Decoder)**: 负责根据编码器提供的中间表示，逐个生成输出序列中的元素。

#### **第二部分：解码器的工作模式**

解码器的核心任务是生成序列。它主要有两种工作模式：自回归（AT）和非自回归（NAT）。

##### **1. 自回归解码器 (Autoregressive, AT)**

这是Transformer以及大多数生成模型的**标准工作模式**。它像人类说话或写作一样，**一个词一个词地生成**，并且下一个词的生成依赖于所有已经生成的词。

*   **工作流程**:
    1.  **起始**: 解码器从一个特殊的**起始符 `[BEGIN]`（或`[SOS]`）**开始。
    2.  **生成第一步**: 将`[BEGIN]`作为解码器的输入，解码器会结合编码器提供的全部信息，通过一个Softmax层输出一个在整个**词汇表 (Vocabulary)** 上的概率分布。这个词汇表包含了所有可能的字符或子词（Subword）。
    3.  **选择与迭代**: 从概率分布中选择概率最高的词（或通过某种采样策略选择一个词），作为输出序列的第一个词。然后，将这个新生成的词**追加到解码器的输入中**，再次进行下一步的预测。
    4.  **终止**: 这个过程不断重复，直到模型生成一个特殊的**结束符 `[END]`（或`[EOS]`）**，或者达到预设的最大长度。

##### **2. 非自回归解码器 (Non-autoregressive, NAT)**

NAT试图打破AT的串行生成模式，以实现**并行化**，从而大幅提升生成速度。

*   **工作流程**:
    1.  **预测长度**: 首先，需要一个额外的**长度预测器 (Length Predictor)** 来预测输出序列大概应该有多长（例如，预测长度为L）。
    2.  **并行输入**: 向NAT解码器的输入端一次性提供L个`[BEGIN]`（或其他占位符）。
    3.  **并行生成**: NAT解码器**一次性地、并行地**为这L个位置生成对应的输出词。
    4.  **截断**: 通常会生成一个偏长的序列，然后忽略第一个出现的`[END]`之后的所有词。

*   **优缺点**:
    *   **优点**:
        *   **速度快**: 因为所有词都是并行生成的，解码速度极快。
        *   **长度可控**: 可以通过调整长度预测器来间接控制输出长度。例如，在语音合成（TTS）中，可以通过给长度预测器的输出乘以一个系数，来方便地调整语速。
    *   **缺点**:
        *   **性能通常更差**: 因为NAT模型在生成某个词时，无法看到它旁边的其他词是什么，这破坏了输出词之间的内部依赖关系。这会导致所谓的**“多模态问题” (Multimodality Problem)**：对于一个输入，可能有多个同样合理的输出（例如，“你好”可以翻译成“Hello”、“Hi”），AT模型可以通过逐步决策来选择一个连贯的路径，而NAT模型在并行生成时很容易产生一个由不同合理路径的片段拼接而成的、整体不连贯的“缝合怪”结果。

##### **3. 交叉注意力 (Cross-Attention) - Encoder与Decoder的桥梁**

这是连接编码器和解码器的关键。它让解码器在生成每一个词时，都能“关注”到输入序列中最相关的部分。Cross-Attention实际上比Self-Attention出现得更早。

*   **工作流程 (在解码器内部)**:
    1.  **Query来自解码器**: 解码器在生成当前词时，其自身的中间状态构成了**查询向量 (Query)**。这个Query代表了“为了生成下一个词，我需要从原始输入中寻找什么信息？”。
    2.  **Key和Value来自编码器**: **键向量 (Key)** 和 **值向量 (Value)** **全部来自于编码器的最终输出**。它们代表了输入序列中每个位置所包含的“可供查询的索引信息”和“实际内容”。
    3.  **计算与加权**: 解码器的Query会和编码器所有的Key进行匹配，计算出注意力分数，经过Softmax后，用这个权重去加权求和编码器所有的Value。
    4.  **结果**: 最终得到的加权和向量，就是一个为当前解码步骤“量身定制”的、融合了输入序列最相关信息的上下文向量。解码器将利用这个向量和自身的已有信息来生成下一个词。

##### **4. 训练技巧与机制**

*   **教师强制 (Teacher Forcing)**: 在**训练**阶段，为了加速收敛并稳定训练过程，我们不使用模型上一步自己生成的（可能错误的）词作为下一步的输入，而是直接使用**标准答案 (Ground Truth)** 中的词作为下一步的输入。这就像有一个老师（Ground Truth）在旁边强制纠正模型的每一步。
*   **拷贝机制 (Copy Mechanism)**: 在某些任务（如文本摘要、对话）中，输出序列有时需要直接“拷贝”输入序列中的某些词（如人名、地名）。拷贝机制允许模型在生成词汇时，除了可以从固定词汇表中选择，还可以直接从输入序列中“复制”一个词作为输出。
*   **引导式注意力 (Guided Attention)**: 在某些任务中，输入和输出序列具有很强的**单调对齐**关系，例如语音识别和TTS（第一个音素对应第一个词，顺序基本一致）。引导式注意力通过增加一个惩罚项，强制要求注意力权重矩阵大致呈对角线分布，从而引导模型学习这种单调对齐关系，提升稳定性和性能。

#### **第三部分：解码策略与评估**

##### **1. 解码策略：如何选择下一个词**

*   **贪心解码 (Greedy Decoding)**: 最简单的方法。在每一步，总是选择当前概率最高的那个词。缺点是容易陷入局部最优，可能因为眼前的一步最优选择，而错过了全局更优的句子。
*   **集束搜索 (Beam Search)**: 贪心解码的一种改进。在每一步，不再只保留概率最高的一个路径，而是保留**概率最高的k个路径（k被称为Beam Size）**。在下一步，会从这k个路径出发，探索所有可能的下一个词，并再次选出全局最优的k个新路径。这是一种在计算成本和搜索完备性之间的折中方案。
*   **随机性采样**: 在某些需要创造性的任务中（如写诗、故事续写、TTS），我们不希望输出总是千篇一律。可以通过一些采样方法（如Top-k采样、Top-p/Nucleus采样）在解码时引入一定的随机性，使得每次生成的结果都可能不同。

##### **2. BLEU Score：评估机器翻译质量**

*   **训练目标 vs. 评估指标**:
    *   **交叉熵 (Cross Entropy)**: 这是模型在**训练**时优化的目标函数。它在**词元级别 (token-level)** 上衡量模型预测的概率分布与真实词的One-hot表示之间的差距。它**是可微分的**。
    *   **BLEU Score**: 这是在**评估**阶段（尤其是机器翻译）衡量生成句子质量的常用指标。它在**句子级别 (sentence-level)** 上，通过比较模型生成的句子和人类翻译的参考句子之间的 **n-gram（n元词组）**重合度来打分。BLEU Score **是不可微分的**，因此不能直接作为训练的损失函数。
    *   **如何优化BLEU？**: 如果想直接优化BLEU这类不可微的指标，可以借助**强化学习 (Reinforcement Learning)**。将生成句子的过程看作一个策略，BLEU Score作为奖励信号，来更新模型参数。

##### **3. 计划采样 (Scheduled Sampling)**

这是一个试图弥合“教师强制”和“真实推理”之间差距的训练策略。
*   **问题**: 训练时使用Teacher Forcing，而测试时模型只能依赖自己之前生成的结果。这种训练和测试之间的不一致（exposure bias）会导致错误累积。
*   **做法**: 在训练的早期阶段，100%使用Teacher Forcing。随着训练的进行，以一定的概率（这个概率会逐渐增加）开始使用模型自己上一步的预测结果作为下一步的输入。这样可以让模型在训练阶段就逐渐适应并学会从自己的错误中恢复。