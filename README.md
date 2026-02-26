# Learning Log

Short notes on what I study and when I update. I add more courses over time.

## Update Log

### Deep Learning from Scratch

- Book: *Deep Learning from Scratch* (ゼロから作るDeep Learning) by Koki Saitoh
- Building neural networks from scratch using **NumPy only** (no PyTorch/TensorFlow)
- **2026-02-22**: Completed Chapters 1-5
  - Ch.1: Python & NumPy fundamentals
  - Ch.2: Perceptron implementation (AND/OR/XOR gates)
  - Ch.3: Neural networks (activation functions, forward propagation)
  - Ch.4: Training algorithms (loss functions, numerical gradients, SGD)
  - Ch.5: Backpropagation (computational graphs, efficient gradient computation)
  - Trained 2-layer network on MNIST: **97% test accuracy**

### MIT/6.S184

- MIT Class 6.S184: *Generative AI With Stochastic Differential Equations*, 2025 
- An Introduction to Flow Matching and Diffusion Models
- **2026-01-24**: Finished Lab One: Simulating ODEs and SDEs
- **2026-02-02**: Finished Lab Two: Flow Matching and Score Matching
- **2026-02-26**: Finished Lab Three: A Conditional Generative Model for Images
  - Created `sdelib/`: Reusable library for flow matching training and sampling

## Repository Structure

```
learning-log/
├── dl-from-scratch/          # Deep Learning from Scratch (NumPy implementations)
├── Hung-yi_Lee/             # Hung-yi Lee's ML courses (2017-2025)
└── mit/6.S184/                  # MIT Generative AI course materials
```