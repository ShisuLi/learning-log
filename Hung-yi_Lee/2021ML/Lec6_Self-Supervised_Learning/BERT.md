# BERT
- ELMo(Embeddings from Language Models)94M
- BERT(Bidirectional Encoder Representations from Transformers)340M
- ERNIE(Enhanced Representation through kNowledge IntEgration)
- BIG BIRD(Transformer for Longer Sequences)
- GPT-2 1542M
- Megatron 8B
- T5 11B
- Turing NLG 17B
- GPT-3 175B
- Switch Transformer 1.6T

## Self-supervised Learning
- BERT: Transformer Encoder+Linear+Softmax+Cross Entropy Loss
- Masked Token Prediction
- Next Sentence Prediction: Yes or no two sequences are consecutive; not helpful.
- SOP: Sentence Order Prediction: swap two consecutive segments or not; more challenging.
- [CLS], [SEP], 


### Downstream Tasks
可能和训练时的填空题没有关系，但可以被微调用于不同的下游任务
Pre-train + Fine-tune--Semi-supervised Learning

#### GLUE(General Language Understanding Evaluation)
- Nine different tasks for evaluating and analyzing natural language understanding systems, including single-sentence tasks, similarity and paraphrase tasks, and inference tasks.


[CLS] - q1, q2(question) - [SEP] - d1, d2, d3(document)

MASS

BART

T5(Transfer Text-to-Text Transformer)
C4(Clossal Clean Crawled Corpus)

### Why does BERT work?
- The tokens with similar meaning have similar embedding.
- Context is considered. Contextualized word embedding.
- You shall know a word by the company it keeps. (Firth, 1957)
- CBOW vs. skip-gram: word embedding

### Multi-lingual BERT
- Pre-trained on 104 languages.
- Cross-lingual Alignment

好的，我们来将这份关于BERT及其相关模型的笔记，进行全面的翻译、扩展和深度讲解，构建一个围绕BERT的、结构化且包含最新知识的完整文档。

---

### **BERT及其时代：预训练语言模型的崛起与演进**

BERT（Bidirectional Encoder Representations from Transformers）的出现，是自然语言处理（NLP）领域的一个分水岭事件。它与GPT、ELMo等模型共同开启了**“预训练-微调”（Pre-train, Fine-tune）**的新范式，将NLP带入了一个由超大模型驱动的全新时代。

#### **第一部分：群星闪耀 - 预训练模型的演进之路**

笔记中列出了一系列里程碑式的模型，它们共同描绘了模型规模和能力不断指数级增长的宏伟蓝图。

*   **ELMo (Embeddings from Language Models, 94M)**: 真正意义上开启了“动态词向量”或“上下文词向量”的时代。它使用双向LSTM，一个词的表示不再是固定的，而是会根据其上下文动态变化。但它的双向是“伪双向”，只是将从左到右和从右到左两个方向的表示进行了简单的拼接。
*   **BERT (Bidirectional Encoder Representations from Transformers, 340M)**: 革命者。它使用Transformer的Encoder结构，并通过**Masked Language Model (MLM)** 任务，首次实现了**真正意义上的“深度双向”表示**。一个词的表示可以在一个深层网络中同时融合其左右两侧的全部上下文信息。
*   **GPT-2 (Generative Pre-trained Transformer 2, 1.5B)**: 通才的雏形。与BERT（Encoder-only）不同，GPT系列使用Transformer的Decoder结构，专注于文本生成。GPT-2展示了大规模单向模型在零样本（Zero-shot）任务上的惊人潜力。
*   **ERNIE (Enhanced Representation through kNowledge IntEgration)**: BERT的改进版，试图将外部的**知识图谱（Knowledge Graph）**信息融入预训练过程，通过对实体、短语级别的Masking，让模型学习到更结构化的知识。
*   **BIG BIRD & Transformer for Longer Sequences**: 针对原始Transformer注意力机制计算复杂度是序列长度平方（$O(n^2)$）的痛点，这类模型通过稀疏注意力、窗口化注意力等方法，将复杂度降低到线性（$O(n)$），使得模型能够处理数千甚至数万长度的超长序列。
*   **Megatron (8.3B), T5 (11B), Turing NLG (17B)**: 标志着模型参数进入“百亿”时代。这不仅仅是量的堆砌，更带来了质的飞跃，即**“涌现能力”（Emergent Abilities）**的出现。
*   **GPT-3 (175B)**: 引爆AIGC的奇点。它在1750亿参数的巨大规模下，展现出了强大的**上下文学习（In-context Learning）**能力，用户只需在Prompt中给出几个范例，模型就能在不进行任何参数微调的情况下完成新任务。
*   **Switch Transformer (1.6T)**: 首次将模型参数扩展到“万亿”级别。它通过**稀疏混合专家（Mixture of Experts, MoE）**架构实现，即在模型中设置多个“专家”网络，对于每个输入，一个路由网络会智能地选择激活其中一小部分专家来进行计算。这使得模型总参数量可以巨大，但单次计算的成本却能保持在可控范围内。

#### **第二部分：BERT的核心 - 自监督学习 (Self-supervised Learning)**

BERT成功的核心在于其巧妙的**自监督学习**策略。自监督学习指的是不依赖于人工标注的数据，而是从无标签数据本身出发，自动地创造出“标签”来进行训练。BERT主要通过以下两个任务实现：

##### **1. 掩码语言模型 (Masked Language Model, MLM)**

这是BERT的灵魂所在，也是它实现真·双向上下文理解的关键。

*   **做法**:
    1.  随机选取句子中15%的词元（Token）。
    2.  在这15%的词元中：
        *   80%的概率，用一个特殊的`[MASK]`标记替换掉它。
        *   10%的概率，用一个随机的其他词替换掉它。
        *   10%的概率，保持原词不变。
*   **目标**: 训练模型去**预测这些被掩盖或篡改的原始词元是什么**。
*   **为什么有效？**:
    *   为了正确预测`[MASK]`位置的词，模型被迫**同时深度地理解其左侧和右侧的全部上下文信息**。例如，在"I went to the [MASK] to withdraw money."中，模型需要看到左边的"went to"和右边的"withdraw money"，才能高概率地预测出`[MASK]`是"bank"。
    *   后两种策略（随机替换和保持不变）是为了缓解训练（输入有`[MASK]`）和微调（输入没有`[MASK]`）之间的不匹配问题，并强迫模型学习每个词自身的表示。

##### **2. 下一句预测 (Next Sentence Prediction, NSP) & 其演进**

这个任务旨在让模型理解句子与句子之间的关系。

*   **原始NSP**:
    *   **做法**: 从文本中抽取句子对(A, B)。50%的情况下，B是A的真实下一句；50%的情况下，B是从语料库中随机抽取的句子。
    *   **目标**: 训练模型判断B是否是A的下一句（一个二分类问题）。
    *   **问题**: 后续研究（如RoBERTa）发现，这个任务过于简单，模型可能更多地是去学习两个句子主题是否相关，而不是它们是否真的“连续”，**因此被证明对下游任务性能提升帮助不大，甚至有害**。

*   **SOP (Sentence Order Prediction)**: ALBERT等模型提出的改进版。
    *   **做法**: 总是从文本中抽取两个**连续**的片段(A, B)。50%的情况下保持(A, B)的顺序，50%的情况下交换为(B, A)。
    *   **目标**: 训练模型判断这两个片段的顺序是否被交换过。
    *   **优势**: 这个任务更具挑战性，它强迫模型去学习更细粒度的语篇连贯性（Discourse Coherence），而不是简单的主题匹配。

##### **特殊标记 (Special Tokens)**

*   `[CLS]`: "Classification"的缩写。每个输入序列的开头都会放置这个标记。BERT的设计意图是，这个标记对应的最终输出向量，可以被视为整个输入序列的**聚合表示**，因此通常用它来接一个分类器，进行句子（对）级别的分类任务。
*   `[SEP]`: "Separator"的缩写。用于分隔两个句子，或在单句输入的末尾作为结束符。

#### **第三部分：从预训练到应用 - 微调范式**

BERT本身只是一个通用的“语言理解引擎”，它并不知道具体要做什么任务。将它应用到下游任务的过程，就是**微调（Fine-tuning）**。

*   **预训练 + 微调 (Pre-train + Fine-tune)**:
    1.  **预训练阶段**: 在海量的无标签数据上进行自监督学习（MLM, SOP等），这个过程耗资巨大，但只需要做一次。
    2.  **微调阶段**: 针对特定的下游任务（如情感分析），在预训练好的BERT模型之上添加一个小的、任务特定的输出层。然后，使用该任务的**有标签数据**来训练模型。在这个过程中，**BERT自身的参数也会被“微调”**，以更好地适应新任务。
    *   **本质**: 这种模式可以被看作是一种**半监督学习（Semi-supervised Learning）**，因为它利用了大量的无标签数据和少量的有标签数据。

*   **GLUE (General Language Understanding Evaluation)**: 一个著名的NLP基准测试集，包含了九项不同的自然语言理解任务，如单句分类、句子对相似度判断、自然语言推断等。它被广泛用于衡量一个预训练模型通用语言理解能力的“期中考试”。

#### **第四部分：为什么BERT有效？**

1.  **上下文相关的动态词向量 (Contextualized Word Embedding)**:
    *   在BERT出现之前，像Word2Vec这样的词嵌入是**静态的**。无论在什么句子中，“bank”这个词的向量都是固定的，无法区分“河岸”和“银行”。
    *   BERT生成的词向量是**动态的**。同一个词“bank”，在“river bank”和“investment bank”这两个不同上下文中，其输出的向量表示是完全不同的。BERT真正实现了语言学中的一句名言：**"You shall know a word by the company it keeps." (观其伴以知其言, Firth, 1957)**。

2.  **深度融合的双向上下文**:
    *   与ELMo的浅层拼接不同，BERT通过多层Transformer Encoder，让每个词的表示都能在深度网络中**反复、充分地**与全局上下文进行信息交互和融合。

3.  **与传统词嵌入方法的联系**:
    *   MLM任务在形式上与Word2Vec中的**CBOW (Continuous Bag-of-Words)** 模型有相似之处（都是根据上下文预测中心词）。但BERT将其置于一个深度的、非线性的Transformer架构中，并引入了真正的双向性，从而获得了远超传统方法的强大表示能力。

#### **第五部分：BERT的扩展与变体**

*   **多语言BERT (Multi-lingual BERT)**:
    *   Google发布了一个在**104种语言**的维基百科上共同预训练的BERT模型。
    *   惊人的发现是，即使没有明确的跨语言训练目标，这个模型也能在不同语言之间学到一定程度的**跨语言对齐（Cross-lingual Alignment）**。例如，英语的“dog”和德语的“Hund”在模型的高维空间中位置会非常接近。这使得我们可以在一种语言（如英语，数据多）上微调模型，然后直接在另一种语言（如斯瓦希里语，数据少）上进行零样本或少样本的预测，并取得不错的效果。

*   **面向生成的预训练 (MASS, BART, T5)**:
    *   **MASS**: 提出了一种用于序列到序列（Seq2Seq）任务的预训练方法，它Mask掉Encoder输入的一个连续片段，并强迫Decoder去预测这个被Mask的片段。
    *   **BART**: 结合了BERT的双向编码器和GPT的自回归解码器。其预训练任务更加多样，例如对原始文本进行各种“破坏”（如Token Masking, Deletion, Sentence Permutation），然后训练模型去**还原（Denoise）**出原始文本。
    *   **T5 (Text-to-Text Transfer Transformer)**: 终极的统一范式。T5将**所有NLP任务都统一为“文本到文本”（Text-to-Text）**的格式。例如，对于分类任务，输入是“classify: a great movie!”，期望的输出文本是“positive”。它在一个巨大的、经过清洗的网页语料库**C4 (Colossal Clean Crawled Corpus)** 上进行预训练，展现了惊人的泛化能力。