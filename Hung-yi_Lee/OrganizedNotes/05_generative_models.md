# 5 生成式模型与潜变量

## 5.1 文本 vs 图像生成范式
- **文本**：自回归逐 token（GPT）。
- **图像/多模态**：通常“一步到位”生成隐向量 $z\sim\mathcal N(0,I)$ 后解码；同一描述对应分布而非唯一像素，需显式采样。

## 5.2 Autoencoder 家族
- **基础 AE**：Encoder 压缩、Decoder 重构，最小化重构误差；可用于压缩/异常检测。
- **去噪 AE**：输入加噪，强迫模型复原干净样本，学鲁棒特征。BERT 的 MLM 是序列去噪 AE。
- **特征解耦**：不同维度承载独立属性，可用于语音转换（内容/音色分离）。
- **离散隐空间 (VQ-VAE)**：码本 + 最近邻量化，隐向量离散化，可无标签聚类。
- **文本作为表示**：Seq2Seq2Seq AE，摘要作隐码，再解码回全文；可加判别器提升可读性。

## 5.3 概率生成模型
- **VAE**：Encoder 输出 $\mu,\sigma$，采样 $z$，KL 约束 $q(z|x)$ 近似 $\mathcal N(0,I)$；损失=重构+KL。
- **Flow-based**：可逆变换 $x\leftrightarrow z$，严格对数似然；输入输出维度相同、结构受限但可精确采样/评估。
- **GAN**：只有生成器（解码器）和判别器对抗；易模式崩溃，信号间接。可与 VAE/Flow/Diffusion 组合。
- **PixelRNN/WaveNet**：自回归像素/音频生成，计算开销大。

## 5.4 Diffusion
- **正向过程**：逐步向数据加噪至各向同性高斯。
- **反向训练**：学习条件去噪器 $D_\theta(x_t,t)$ 估计噪声或得分函数。
- **采样**：从纯噪声迭代去噪生成图像；可结合文本条件、Classifier/Guidance、与 GAN 等混合。

## 5.5 生成模型对比
- VAE/Flow/Diffusion：Encoder-Decoder 结构（Flow 的 Encoder 可逆）；GAN 无编码器。
- 质感：GAN 细节锐利但难训；VAE 模糊；Flow 可精确但重；Diffusion 质量高、采样慢。
