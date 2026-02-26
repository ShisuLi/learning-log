# MIT 6.S184: Generative AI With Stochastic Differential Equations (2025)

This repository contains my learning materials and implementations for **MIT Class 6.S184: Generative AI With Stochastic Differential Equations**, taught in 2025.

## Course Information

- **Title**: An Introduction to Flow Matching and Diffusion Models
- **Official Website**: [https://diffusion.csail.mit.edu/2025/index.html](https://diffusion.csail.mit.edu/2025/index.html)
- **Lecture Notes**: [arXiv:2506.02070](https://arxiv.org/abs/2506.02070)
- **Instructors**: Peter Holderrieth and Ezra Erives

### Citation

```bibtex
@misc{flowsanddiffusions2025,
  author       = {Peter Holderrieth and Ezra Erives},
  title        = {Introduction to Flow Matching and Diffusion Models},
  year         = {2025},
  url          = {https://diffusion.csail.mit.edu/}
}
```

## Repository Structure

```
.
├── 01_slides/          # Course lecture slides
├── 02_tutorial/        # Tutorial materials and exercises
├── 03_paper/           # Related research papers
├── 04_notes/           # Personal study notes
├── 05_labs/            # Laboratory assignments and implementations
│   ├── lab01/          # Lab 1: Introduction to SDEs
│   ├── lab02/          # Lab 2: Flow Matching
│   ├── lab03/          # Lab 3: Conditional Generation with CFG
│   └── sdelib/         # ⭐ Production-ready flow matching library
└── main.py             # Main entry point
```

### SDELib - Production Library

The [05_labs/sdelib](05_labs/sdelib) directory contains a **production-ready extraction** of the essential components from the lab materials. This is a minimal, efficient library suitable for real model training:

- **Compact**: ~1,300 LOC (vs ~3,800 in lab materials)
- **Focused**: Only essential, reusable components
- **Optimized**: Production-ready code without educational overhead
- **Clean API**: Well-documented, type-hinted interfaces

See [sdelib/README.md](05_labs/sdelib/README.md) for usage and [sdelib/COMPARISON.md](05_labs/sdelib/COMPARISON.md) for differences from lab materials.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

Install `uv`:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Install Python 3.12 (if not already installed)
uv python install 3.12

# Create virtual environment
uv venv

# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies (automatically handled by uv)
uv sync

# Install Jupyter kernel for notebooks
uv run python -m ipykernel install --user --name=6.S184
```

### Running Notebooks

```bash
# Start Jupyter Lab
uv run jupyter lab
```

Or open the notebooks directly in VS Code with the Jupyter extension.

## License

This is a personal learning repository. Please refer to the original course materials for their respective licenses.