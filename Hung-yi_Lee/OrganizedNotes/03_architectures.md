# 3 模型结构与序列建模

## 3.1 CNN：局部性与参数共享
- **假设**：模式局部且平移不变。
- **卷积层**：感受野小（如 3x3xC），跨位置共享权重，得到特征图；多滤波器并行检测不同模式。
- **池化**：降采样与小范围平移不变性（常用 Max Pooling）；可被步长卷积替代。
- **优劣**：参数少、对图像/棋盘等局部结构友好；但对旋转/缩放不自带不变性，需要数据增强。

## 3.2 RNN / LSTM / GRU
- **RNN**：共享权重沿时间展开，易梯度消失/爆炸；可用 ReLU+单位矩阵初始化、梯度裁剪缓解。
- **LSTM**：引入细胞状态 $C_t$ 与三门：
  - 遗忘门 $f_t$ 控制保留旧记忆；
  - 输入门 $i_t$ + 候选记忆 $\tilde C_t$ 写入新信息；
  - 输出门 $o_t$ 决定输出 $h_t=o_t\odot\tanh(C_t)$。
  解决长依赖，参数约为简单 RNN 4 倍。
- **GRU**：更新门/重置门，合并细胞与隐藏状态，参数更少。
- **BiRNN**：正反向结合，获取全局上下文。
- **高级应用**：CTC 解决输入输出长度未知对齐；Seq2Seq/Attention/CRF 组合；Neural Turing Machine 外部存储；Seq2Seq+Attention 用于翻译/语音等。

## 3.3 自注意力与 Transformer
- **Self-Attention 三步**：$q=W^q a,\ k=W^k a,\ v=W^v a$；得分 $\alpha=q\cdot k$（或加性），Softmax 归一；输出 $b=\sum_i \hat\alpha_i v_i$。可并行计算全序列，捕捉长依赖。
- **多头注意力**：多组 $(W^q,W^k,W^v)$ 生成 $b^{(h)}$，拼接后经 $W^O$ 融合。
- **位置编码**：弥补排列不变性，可用正弦/余弦手工或可学习向量，与输入相加。
- **复杂度**：$O(L^2)$，长序列可用截断/稀疏注意力。
- **与 CNN 对比**：自注意力视作动态、可学习感受野的“全局 CNN”；CNN 可视作局部、自共享的简化注意力。

## 3.4 Seq2Seq（Encoder-Decoder）
- **编码器**：常为 Transformer Encoder（自注意力+FFN+残差+LayerNorm）；输入=嵌入+位置编码。
- **解码器 AR（Autoregressive）**：以 `[BEGIN]` 启动，逐步生成 token，常用 Cross-Attention 读取编码器输出；Teacher Forcing 训练；终止于 `[END]`。
- **解码器 NAR（Non-autoregressive）**：并行生成，需长度预测；速度快但易多模态缝合、质量略差。
- **Cross-Attention**：Query 来自解码器，Key/Value 来自编码器，得到对输入的动态对齐。
- **辅助机制**：复制机制（可从输入直接拷贝 token）、单调/位置引导注意力（语音/TTS）、CTC 对齐。

## 3.5 解码与评估
- **Beam Search**：保留前 k 条路径，平衡搜索质量与成本；贪心为 k=1。
- **随机采样**：Top-k/Top-p 适合创意生成。
- **Scheduled Sampling**：训练中逐步用模型输出替换真值输入，缓解曝光偏差。
- **BLEU**：n-gram 重合评估翻译，训练可用 RL 优化；损失仍用 token 级交叉熵。
