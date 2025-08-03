# TensorWeaver

<p align="center">
  <img src="docs/assets/logo.png" alt="TensorWeaver Logo" width="200"/>
</p>

<p align="center">
  <strong>🧠 Finally understand how PyTorch really works</strong><br>
  <em>Build your own deep learning framework from scratch</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/tensorweaver/"><img src="https://img.shields.io/pypi/v/tensorweaver.svg" alt="PyPI version"></a>
  <a href="https://github.com/howl-anderson/tensorweaver/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/howl-anderson/tensorweaver/stargazers"><img src="https://img.shields.io/github/stars/howl-anderson/tensorweaver.svg" alt="GitHub stars"></a>
  <a href="https://www.tensorweaver.ai"><img src="https://img.shields.io/badge/docs-tensorweaver.ai-blue" alt="Documentation"></a>
</p>

---

## 🤔 **Ever feel like PyTorch is a black box?**

```python
# What's actually happening here? 🤷‍♂️
loss.backward()  # Magic? 
optimizer.step()  # More magic?
```

**You're not alone.** Most ML students and engineers use deep learning frameworks without understanding the internals. That's where TensorWeaver comes in.

## 🎯 **What is TensorWeaver?**

TensorWeaver is the **educational deep learning framework** that shows you exactly how PyTorch works under the hood. Built from scratch in pure Python, it demystifies automatic differentiation, neural networks, and optimization algorithms.

> **Think of it as "PyTorch with the hood open"** 🔧

### **🎓 Perfect for:**
- **CS Students** learning machine learning internals
- **Self-taught developers** who want to go beyond tutorials  
- **ML Engineers** debugging complex gradient issues
- **Educators** teaching deep learning concepts
- **Curious minds** who refuse to accept "magic"

> **💡 Pro Tip**: Use `import tensorweaver as torch` for seamless PyTorch compatibility!

## ⚡ **Quick Start - See the Magic Yourself**

```bash
pip install tensorweaver
```

```python
import tensorweaver as torch  # PyTorch-compatible API!

# Build a neural network (just like PyTorch!)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 128)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

model = SimpleModel()

# Train it
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# The difference? You can see EXACTLY what happens inside! 👀
```

🚀 **[Try it live in your browser →](https://mybinder.org/v2/gh/howl-anderson/tensorweaver/HEAD?urlpath=%2Fdoc%2Ftree%2Fmilestones%2F01_linear_regression%2Fdemo.ipynb)**

## 🧠 **What You'll Learn**

<table>
<tr>
<td width="50%">

### **🔬 Deep Learning Internals**
- How automatic differentiation works
- Backpropagation step-by-step
- Computational graph construction
- Gradient computation and flow

</td>
<td width="50%">

### **🛠️ Framework Design**
- Tensor operations implementation
- Neural network architecture
- Optimizer algorithms
- Model export (ONNX) mechanisms

</td>
</tr>
</table>

## 💎 **Why TensorWeaver?**

| 🏭 **Industrial Frameworks** | 🎓 **TensorWeaver** |
|------------------------------|---------------------|
| ❌ Complex C++ codebase | ✅ Pure Python - readable by humans |
| ❌ Optimized for performance | ✅ Optimized for learning |
| ❌ "Trust us, it works" | ✅ "Here's exactly how it works" |
| ❌ Intimidating for beginners | ✅ Designed for education |

### **🚀 Key Features**

- **🔍 Transparent Implementation**: Every operation is visible and understandable
- **🐍 Pure Python**: No hidden C++ complexity - just NumPy and Python
- **🎯 PyTorch-Compatible API**: Same interface, easier transition
- **📚 Educational Focus**: Built for learning, not just using
- **🧪 Complete Functionality**: Autodiff, neural networks, optimizers, ONNX export
- **📖 Growing Documentation**: Clear explanations with working examples

## 🎮 **Learning Path**

### **🌱 Beginner Level**
1. **[Tensor Basics](milestones/01_linear_regression/)** - Understanding tensors and operations
2. **[Linear Regression](milestones/01_linear_regression/demo.ipynb)** - Your first neural network
3. **Automatic Differentiation** - How gradients are computed *(coming soon)*

### **🌿 Intermediate Level**  
4. **[Multi-layer Networks](milestones/03_multilayer_perceptron/)** - Building deeper models
5. **Loss Functions & Optimizers** - Training dynamics *(coming soon)*
6. **[Model Export](milestones/02_onnx_export/)** - ONNX export and deployment

### **🌳 Advanced Level**
7. **Custom Operators** - Extending the framework *(coming soon)*
8. **Performance Optimization** - Making it faster *(coming soon)*
9. **GPU Support** - Scaling computations *(in development)*

> **📝 Note**: Some documentation links are still in development. Check our [milestones](milestones/) for working examples!

## 🎯 **Quick Examples**

<details>
<summary><b>🔬 See Automatic Differentiation in Action</b></summary>

```python
import tensorweaver as torch

# Create tensors
x = torch.tensor([2.0])
y = torch.tensor([3.0])

# Forward pass
z = x * y + x**2
print(f"z = {z.data}")  # [10.0]

# Backward pass - see the magic!
z.backward()
print(f"dz/dx = {x.grad}")  # [7.0] = y + 2*x = 3 + 4  
print(f"dz/dy = {y.grad}")  # [2.0] = x
```

</details>

<details>
<summary><b>🧠 Build a Neural Network from Scratch</b></summary>

```python
import tensorweaver as torch

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Every operation is transparent!
model = MLP()
print(model)  # See the architecture
```

</details>

## 🎯 **Why Students Love Learning This Way**

Instead of mysterious "black box" operations, TensorWeaver shows you:
- **Transparent code** - Every function is readable Python
- **Step-by-step execution** - See exactly how gradients flow
- **PyTorch compatibility** - Easy transition to production frameworks
- **Educational focus** - Built for understanding, not just using

*Real testimonials coming as the community grows!*

## 🚀 **Get Started Now**

### **📦 Installation**
```bash
# Option 1: Install from PyPI (recommended)
pip install tensorweaver

# Option 2: Install from source (for contributors)
git clone https://github.com/howl-anderson/tensorweaver.git
cd tensorweaver
poetry install
```

### **🎯 Your Learning Journey Starts Here**

1. **[🧪 Try Examples](milestones/)** - Hands-on Jupyter notebooks  
2. **[🎮 Interactive Playground](https://mybinder.org/v2/gh/howl-anderson/tensorweaver/HEAD)** - No setup required
3. **[💬 Join Community](https://github.com/howl-anderson/tensorweaver/discussions)** - Ask questions and share projects
4. **[📖 Read Documentation](https://tensorweaver.ai)** - Framework overview *(expanding soon)*

## 🤝 **Contributing**

TensorWeaver thrives on community contributions! Whether you're:
- 🐛 **Reporting bugs**
- 💡 **Suggesting features** 
- 📖 **Improving documentation**
- 🧪 **Adding examples**
- 🔧 **Writing code**

We welcome you! Please open an issue or submit a pull request - contribution guidelines coming soon!

## 📚 **Resources**

- **📖 [Documentation](https://tensorweaver.ai)** - Framework overview
- **💬 [Discussions](https://github.com/howl-anderson/tensorweaver/discussions)** - Community Q&A
- **🐛 [Issues](https://github.com/howl-anderson/tensorweaver/issues)** - Bug reports and feature requests
- **📧 [Follow Updates](https://github.com/howl-anderson/tensorweaver)** - Star/watch for latest changes

## 🎓 **Educational Use**

Using TensorWeaver in your course? We'd love to help!

- **🎯 [Working Examples](milestones/)** - Ready-to-use Jupyter notebooks
- **💬 [Get Support](https://github.com/howl-anderson/tensorweaver/discussions)** - Ask questions and get help
- **📧 [Contact Us](https://github.com/howl-anderson/tensorweaver/issues)** - Let us know about your educational use case

*Curriculum materials and instructor resources are in development - please reach out if you're interested!*

## ⭐ **Why Stars Matter**

If TensorWeaver helped you understand deep learning better, please consider starring the repository! It helps others discover this educational resource.

<p align="center">
  <a href="https://github.com/howl-anderson/tensorweaver/stargazers">
    <img src="https://img.shields.io/github/stars/howl-anderson/tensorweaver?style=social" alt="GitHub stars">
  </a>
</p>

## 📄 **License**

TensorWeaver is MIT licensed. See [LICENSE](LICENSE) for details.

## 🙏 **Acknowledgments**

- Inspired by educational frameworks: **Micrograd**, **TinyFlow**, and **DeZero**
- Thanks to the PyTorch team for the API design
- Grateful to all contributors and the open-source community

---

<p align="center">
  <strong>Ready to peek behind the curtain?</strong><br>
  <a href="https://tensorweaver.ai">🚀 Start Learning at tensorweaver.ai</a>
</p>