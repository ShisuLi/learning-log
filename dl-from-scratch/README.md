# Deep Learning from Scratch

Learning neural networks from first principles by implementing everything with **NumPy only** (no PyTorch/TensorFlow). Based on the book "Deep Learning from Scratch" (ゼロから作る Deep Learning
by Koki Saitoh).

## Progress

- [x] **Chapter 1**: Python & NumPy fundamentals
- [x] **Chapter 2**: Perceptron (AND/OR/XOR gates)
- [x] **Chapter 3**: Neural Networks (activation functions, forward propagation)
- [x] **Chapter 4**: Training (loss functions, numerical gradients, SGD)
- [x] **Chapter 5**: Backpropagation (computational graphs, efficient gradients)
- [ ] **Chapter 6**: Optimization techniques
- [ ] **Chapter 7**: Convolutional Neural Networks (CNN)
- [ ] **Chapter 8**: Deep Learning applications

## Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv init --python 3.12
uv venv
uv add numpy matplotlib seaborn scikit-learn
uv add --dev ruff pytest jupyterlab ipykernel
uv run python -m ipykernel install --user --name=dl-from-scratch
```

## Project Structure

```
dl-from-scratch/
├── notebooks/          # Learning notebooks (Ch.1-5 completed)
│   ├── 01_python_fundamental.ipynb
│   ├── 02_perceptron.ipynb
│   ├── 03_neural_network.ipynb
│   ├── 04_training.ipynb
│   └── 05_back_propagation.ipynb
├── common/            # Reusable neural network components
│   ├── functions.py   # Activation & loss functions
│   ├── gradient.py    # Numerical differentiation
│   └── layers.py      # Network layers (Affine, ReLU, etc.)
├── dataset/           # MNIST data loader
│   └── mnist.py
├── models/            # Saved model parameters (.pkl files)
└── images/           # Diagrams for notebooks
```

## References

- Book: ["Deep Learning from Scratch" by Koki Saitoh](https://www.oreilly.co.jp/books/9784873117584/)
- GitHub: [ゼロから作る Deep Learning
](https://github.com/oreilly-japan/deep-learning-from-scratch)