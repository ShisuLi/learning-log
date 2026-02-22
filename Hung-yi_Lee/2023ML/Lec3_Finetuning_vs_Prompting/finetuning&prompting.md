GPT
BERT

期待大语言模型变成专才：翻译、摘要，解决某一特定任务
变成通才：用人类自然语言的**prompt**来要求模型执行翻译、摘要等任务

所有自然语言处理的问题都是问答问题

成为专才的好处：专才在单一任务上有机会赢过通才
成为通才的好处：只要重新设计prompt就可以快速开发新功能

BERT改装，用于单一句子分类、标注，成对句子分类，问答等任务
- 微调预训练语言模型：语言模型作为初始化参数，update network parameters by gradient descent for specific tasks--翻译专才
- 语言模型插入Adapter额外模组，语言模型本身参数不动，update parameters in the adapter
adapterhub.ml
Bitfit, houlsby, adapterbias, prefix-tuning, LoRA
- 只更新少量参数，节省存储空间和计算资源

一百个任务存一百个大模型——一个大模型+一百个adapter插件 efficient finetuning

GPT没有跟BERT一样去做各种微调参数 
进一步成为通才：
提供题目叙述：Instruction learning
T0：Multi-task training--Zero-shot generalization
FLAN: Performance on unseen task types 
提供题目范例：In-context learning，有论文认为是启动模型做本就有能力做的特定任务，而不是从例子中学习；也有论文说in-context learning能达到gradient descent的效果；有论文表面，大的模型才能从context learning中进行真正的学习，并且可能从错误的示例中进行错误的学习，甚至可以直接作为分类器即输入向量输出分类

Chain of Thought (CoT) Prompting
Few shot vs. Few shot CoT
Zero shot vs. Zero shot CoT: Let's think step by step.

self-consistency: sample a diverse set of reasoning paths, marginalize out reasoning paths to aggregate final answers, 参考信心分数给得分设置权重

Least to Most Prompting：通过In context learning教会模型做problem reduction，再依次解决subproblem

用机器来找prompt
Hard Prompt：discrete 文字
Soft Prompt：continuous 向量（Trained by task-specific labeled data）+文字，相当于Adapter at the input
Using reinforcement learning: 根据response-Reward Function 来update Generator
Using an LM to find prompt: 1。 LLMs as inference models: forward generation template，给输入输出，让模型找到prompt;2. LLMs as scoring models: 分别尝试这些prompt的效果并打分；3. (Optional)LLMs as Resampling Models:对打分的high score candidates生成变体得到similar candidates，再次评分: Let's work this out in a step by step way to be sure we have the right answer.

好的，我们来将这份笔记扩展成一份全面、细致且反映最新进展的中文大语言模型（LLM）知识图谱。这份扩展版将不仅涵盖笔记中的要点，还会补充其背后的原理、发展脉络和前沿趋势。

---

### **大语言模型（LLM）的演进：从专才到通才的范式革命**

大语言模型的发展经历了两个截然不同的时代，代表着两种核心哲学：**成为特定任务的“专才” (Specialist)** 和 **成为无所不能的“通才” (Generalist)**。这个转变重塑了整个自然语言处理（NLP）领域。

#### **第一阶段：BERT时代 - 微调范式下的“专才”模型 (Fine-tuning Paradigm)**

以BERT（Bidirectional Encoder Representations from Transformers）为代表的模型，开启了“**预训练 + 微调**”的黄金时代。

*   **核心思想**:
    1.  **预训练 (Pre-training)**: 在海量的无标签文本（如维基百科、书籍）上，通过“完形填空”（Masked Language Model, MLM）和“下一句预测”（Next Sentence Prediction, NSP）等任务，让模型学习通用的语言知识，理解语法、语义和基本的世界常识。这个阶段完成后，我们得到了一个强大的、但尚未针对任何具体任务的“语言理解引擎”。
    2.  **微调 (Fine-tuning)**: 针对一个特定的下游任务（如情感分析、文本翻译），在预训练好的模型之上添加一个小的、任务特定的分类头（Task-specific Head），然后用该任务的**有标签数据**来**更新和调整整个模型的所有参数**。

*   **BERT如何成为“专才”**:
    通过微调，BERT可以被“改装”成各种高效的专才模型：
    *   **单句分类**: 在BERT输出的`[CLS]`特殊标记对应的向量后接一个分类器，用于情感分析、新闻分类等。
    *   **序列标注**: 对BERT输出的每个词的向量都接一个分类器，用于命名实体识别（NER）、词性标注等。
    *   **成对句子分类**: 将两个句子打包输入BERT，用`[CLS]`向量判断它们的关系，如语义相似度匹配、自然语言推断（NLI）。
    *   **问答 (Extractive QA)**: 输入问题和文章，BERT负责在文章中找到答案的起始和结束位置。

*   **“专才”范式的好处**:
    *   **性能卓越**: 在特定的、有充足标注数据的任务上，经过微调的专才模型通常能达到极高的性能，甚至在当时可能超过通才模型。

*   **“专才”范式的困境与反思**:
    *   **高昂的成本**: “一百个任务，一百个模型”。每当有一个新任务，就需要收集标注数据，并重新微调、存储一个完整的大模型副本。这在存储、计算和维护上都是巨大的负担。
    *   **参数效率低**: 微调会更新所有（数亿甚至数十亿）参数，而研究发现，可能只需要调整其中一小部分参数就足以适应新任务。

---

#### **参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT) - “专才”的进化**

为了解决微调的困境，学术界提出了PEFT，其核心思想是：**冻结预训练模型的主体参数，只训练少量额外的、可插拔的参数**。这就像给一个全能机器人安装一个特定任务的“插件”。

*   **代表性技术**:
    *   **Adapter Tuning**: 在Transformer的每个层块中插入小型的、瓶颈状的“适配器”模块（Adapter）。训练时只更新这些适配器模块的参数。著名的`AdapterHub.ml`就是一个分享和管理这些适配器插件的平台。
    *   **LoRA (Low-Rank Adaptation)**: 核心洞察是，模型在适应新任务时，其参数矩阵的“变化量”通常是低秩（Low-Rank）的。LoRA通过训练两个小的低秩矩阵来模拟这个变化量，从而将需要更新的参数量减少几个数量级。这是目前最流行和高效的PEFT方法之一。
    *   **Prefix-Tuning / P-Tuning**: 不改变模型内部结构，而是在输入层前加入可训练的、连续的向量（virtual tokens或prompt embedding）。这相当于在“提问”的层面进行优化，让模型更好地理解任务。
    *   **BitFit**: 一种极简的方法，只更新模型中的偏置（bias）参数。

*   **PEFT的优势**:
    *   **高效**: 只需训练和存储极少量的参数（通常不到原模型的1%），极大地节省了计算和存储资源。现在可以实现**“一个大模型 + 一百个任务插件”**的模式。
    *   **避免灾难性遗忘**: 由于主体模型被冻结，它在预训练阶段学到的通用知识不会在适应新任务时被“冲刷”掉。

---

#### **第二阶段：GPT-3及以后 - 提示范式下的“通才”模型 (Prompting Paradigm)**

以GPT（Generative Pre-trained Transformer）系列为代表的模型，尤其是GPT-3的出现，彻底改变了游戏规则。它们没有像BERT那样针对特定任务进行结构改造和参数微调，而是走向了**成为“通才”的道路**，致力于通过**自然语言提示（Prompt）**来直接解决一切问题。

*   **核心思想**:
    **将所有NLP问题都转化为“问答”或“文本生成”问题。** 模型本身不做任何改变，我们通过精心设计的Prompt来引导模型执行特定任务。

*   **“通才”范式的好处**:
    *   **零成本开发新功能**: 想要模型执行一个新任务？不需要收集数据，不需要训练，只需要**重新设计你的Prompt**。这使得新功能的开发和迭代速度呈指数级提升。
    *   **涌现能力 (Emergent Abilities)**: 当模型规模大到一定程度（如超过千亿参数），它们会“涌现”出在小模型上完全不存在的能力，如上下文学习、思维链推理等。

#### **激发“通才”潜能的关键技术：提示工程 (Prompt Engineering)**

如何与一个通才LLM高效沟通？核心就是设计好的Prompt。

1.  **指令微调 (Instruction-tuning)**:
    *   **思想**: 在海量预训练之后，额外使用一个包含了特定指令的混合数据集对模型进行微调。这个数据集涵盖了成百上千种不同的任务，每个任务都以自然语言指令的形式呈现。
    *   **例子**: 数据集中可能包含：“请将‘我爱北京’翻译成英文” ，以及“请总结以下文章：...” 。
    *   **效果**: 经过指令微调的模型，能够更好地理解人类的指令，并泛化到**从未见过的任务类型**上，实现惊人的**零样本（Zero-shot）**能力。
    *   **代表性工作**: `T0`, `FLAN`等模型证明了多任务指令微调能显著提升模型在未知任务上的表现。

2.  **上下文学习 (In-context Learning, ICL)**:
    *   **思想**: 在Prompt中为模型提供几个任务的**示例（Demonstrations）**，模型就能“照猫画虎”，在不更新任何参数的情况下解决新的问题。这被称为**少样本（Few-shot）**学习。
    *   **例子**:
        ```
        Prompt:
        把英文翻译成中文。
        sea otter -> 海獭
        peppermint -> 薄荷
        plush girafe -> 长颈鹿毛绒玩具
        cheese -> ? 
        ```
        模型会直接输出“奶酪”。
    *   **原理争议**:
        *   **能力触发论**: 一种观点认为，模型并非真的从这几个例子中“学习”，而是这些例子作为一种“任务定位信号”，激活了模型在预训练中早已学到的、但处于“休眠”状态的相关能力。
        *   **隐式梯度下降论**: 另一篇有影响力的论文指出，上下文学习在深层次上与通过梯度下降更新模型参数具有等效的数学形式。
        *   **规模依赖论**: 研究表明，只有**足够大的模型**才能真正从上下文中进行有效的学习。小模型无法展现出这种能力。同时，大模型也可能从**错误的示例**中进行错误的“学习”，直接模仿错误的模式。

3.  **思维链提示 (Chain-of-Thought, CoT) Prompting**:
    *   **思想**: 对于需要多步推理的复杂问题（如数学应用题），直接让模型输出答案往往会出错。CoT的核心是**引导模型在输出最终答案前，先输出详细的、一步步的推理过程**。
    *   **例子**:
        *   **标准Few-shot**: `问题：... 答案：...`
        *   **Few-shot CoT**: `问题：... 推理步骤：...所以答案是...`
    *   **Zero-shot CoT**: 一个惊人的发现是，即使不提供任何示例，只需在Prompt末尾加上一句简单的“魔法咒语”——**“Let's think step by step.” (让我们一步一步地思考。)**，就能显著提升模型在复杂推理任务上的表现。

4.  **高级推理策略**:
    *   **自洽性 (Self-Consistency)**: 不只让模型推理一次，而是多次（例如，通过设置较高的`temperature`参数）生成**多个不同的推理路径**。然后，通过投票等方式，选择出现次数最多的那个答案作为最终答案。这就像让多个“思考者”独立思考后达成共识，结果更可靠。
    *   **由少至多提示 (Least-to-Most Prompting)**: 针对更复杂的问题，先通过上下文学习教会模型如何将一个大问题**分解成一系列子问题**（Problem Reduction），然后再引导模型**依次解决**这些子问题，并将前一个子问题的答案作为后一个子问题的输入。

#### **自动化提示工程：让机器自己寻找最佳Prompt**

手动设计Prompt费时费力，于是研究者开始探索如何让机器自动完成这项工作。

1.  **Prompt的分类**:
    *   **硬提示 (Hard Prompt)**: 人类可读、可编辑的**离散文本**。我们通常所说的Prompt就是指这个。
    *   **软提示 (Soft Prompt)**: 人类不可读的**连续向量**。它相当于在模型的输入端加入了一个可训练的“适配器”，通过任务相关的标注数据来优化这些向量，使其成为该任务的最佳“引导信号”。

2.  **自动化寻找硬提示的方法**:
    *   **基于强化学习 (RL)**: 将LLM本身看作一个**生成器 (Generator)**，它生成Prompt。然后用这个Prompt去解决任务，根据任务的完成情况（如准确率）得到一个**奖励 (Reward)**。再用这个奖励信号，通过强化学习算法来更新Prompt生成器，使其倾向于生成能获得更高奖励的Prompt。
    *   **基于LLM自身 (In-context)**: 利用LLM强大的文本生成和理解能力来寻找Prompt。
        1.  **LLM作为推理模型**: 给LLM几个输入和输出的范例，然后让它**反向生成**可能产生这些结果的指令模板（Prompt）。
        2.  **LLM作为评分模型**: 准备一堆候选的Prompt，让LLM分别用它们去解决任务，然后根据结果的好坏给每个Prompt打分。
        3.  **LLM作为重采样/变体生成模型**: 对那些得分高的Prompt，让LLM生成一些措辞不同但意思相近的**变体**，扩大候选池，然后再次进行评分，不断迭代优化。例如，可能会从“Let's think step by step”变体出“Let's work this out in a step-by-step way to be sure we have the right answer.”这样的更优版本。

---

### **总结：两种范式的融合与未来**

当今的LLM发展呈现出一种融合趋势：
*   **以“通才”为基座**: 一个经过海量数据预训练和指令微调的、能力极强的“通才”基础模型是所有应用的核心。
*   **以“专才”插件为增强**: 当需要在特定领域（如医疗、法律）达到极致性能或保证事实准确性时，通过PEFT（如LoRA）或领域数据继续微调，来创建高效的“专才”插件。

这种“**通用大模型 + 专用小插件**”的模式，兼顾了通才的灵活性和专才的高性能，正在成为LLM落地应用的主流路径，也预示着AI技术与人类社会更深度融合的未来。