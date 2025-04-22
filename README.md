# TensorWeaver

<p align="center">
  <img src="docs/assets/logo.png" alt="TensorWeaver Logo" width="200"/>
</p>

<p align="center">
  <strong>A modern educational deep learning framework for students, engineers and researchers</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#license">License</a> •
  <a href="#acknowledgments">Acknowledgments</a>
</p>

## Introduction

TensorWeaver is a deep learning framework designed specifically for students, engineers and researchers who want to understand how deep learning frameworks work under the hood. Unlike industrial frameworks like PyTorch and TensorFlow that prioritize performance and scalability, TensorWeaver focuses on clarity, readability, and simplicity.

Built entirely in Python with only NumPy as a dependency, TensorWeaver's codebase is transparent and approachable, making it an ideal learning resource for those who want to demystify the "magic" behind modern AI frameworks.

The target users are students, engineers and researchers who want to fully understand the working principles of deep learning frameworks and gain the skills to debug, extend and optimize their own projects with any deep learning framework.

## Features

- **Purely Educational**: Designed from the ground up as a learning tool with clear, well-documented code
- **PyTorch-compatible API**: Exactly the same interface as PyTorch, so you can easily switch between TensorWeaver and PyTorch
- **Lightweight and Readable**: Built with pure Python and minimal dependencies (merely NumPy)
- **Fully Functional**: Supports essential deep learning components:
  - Automatic differentiation engine
  - Common neural network operators
  - Loss functions and optimizers
  - Model definition and training
- **Advanced Capabilities**:
  - ONNX export functionality
- **Comprehensive Documentation**: Detailed explanations of implementation details and design choices

## PyTorch Compatibility

TensorWeaver is designed to be compatible with PyTorch 2.6, allowing for an easier learning transition between frameworks.

## PyTorch API Coverage

TensorWeaver targets coverage of 20% of the most commonly used PyTorch APIs, following the 80/20 principle (where 20% of features cover 80% of use cases). The specific API list has not been compiled yet and will be determined once appropriate data sources or statistical tools are identified for analyzing API usage patterns.

## Online demo

<a href="https://mybinder.org/v2/gh/howl-anderson/tensorweaver/HEAD?urlpath=%2Fdoc%2Ftree%2Fmilestones%2F01_linear_regression%2Fdemo.ipynb" target="_blank">Click here to launch the online demo</a>

## Installation

```bash
# Install from PyPI
pip install tensorweaver

# Or install from source
git clone https://github.com/howl-anderson/tensorweaver.git
cd tensorweaver
poetry install
```

see [poetry](https://python-poetry.org/docs/#installation) for more details if you don't have poetry installed.

## Examples

See [milestones](milestones/) for examples.


## Documentation

see [https://www.tensorweaver.ai](https://www.tensorweaver.ai)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The project draws inspiration from educational frameworks like Micrograd, TinyFlow, and DeZero
- Special thanks to the open-source deep learning community for their pioneering work
- Thanks to all contributors and users who help improve this educational resource