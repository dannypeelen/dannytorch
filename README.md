<h1>DannyTorch</h1>

DannyTorch is a minimalist deep learning framework built from scratch in Python by <b>Danny Peelen</b> — no PyTorch, just math and NumPy.

## Overview

DannyTorch is an educational yet fully functional deep learning library implementing automatic differentiation, neural network modules, and optimizers from first principles. It was built specifically to deeply understand what happens underneath frameworks like PyTorch.

## Features

- **Autograd engine** — scalar and tensor-level reverse-mode automatic differentiation with a dynamic computational graph
- **`nn` module** — `Linear`, `MLP`, `LayerNorm`, `ModuleList`, and more
- **Loss functions** — `CrossEntropyLoss` (with fused softmax + Jacobian), `MSELoss`
- **Optimizers** — `SGD`, `Adam` (in `optim.py`)
- **LLM primitives** — `masked_fill`, `chunk`, batched matmul, `exp`, `sum` for transformer support

## Installation

```bash
pip install git+https://github.com/dannypeelen/dannytorch.git
```

Or for development:

```bash
git clone https://github.com/dannypeelen/dannytorch.git
cd dannytorch
pip install -e .
```

## Quick Start

Feel free to look at ```test.py``` and run various tests using DannyTorch, or build your own!

## Motivation

Built to answer the question: *what is PyTorch actually doing?* If you want to understand deep learning beyond the API surface, building it yourself is the fastest path.

Inspired by [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/geohot/tinygrad). Was aiming for a nice in-between.

## License

MIT