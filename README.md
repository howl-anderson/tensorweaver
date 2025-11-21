# TensorWeaver

<p align="center">
  <img src="docs/assets/logo.png" alt="TensorWeaver Logo" width="200"/>
</p>

<p align="center">
  <strong>🧠 A transparent, debuggable deep learning framework</strong><br>
  <em>PyTorch-compatible implementation with full visibility into internals</em>
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

TensorWeaver is a **transparent deep learning framework** that reveals exactly how PyTorch works under the hood. Built from scratch in pure Python, it provides complete visibility into automatic differentiation, neural networks, and optimization algorithms.

> **Think of it as "PyTorch with full transparency"** 🔧

### **🎓 Perfect for:**
- **ML Engineers** debugging complex gradient issues and understanding framework internals
- **Researchers** who need full control over their implementations
- **Software Engineers** building custom deep learning solutions
- **Technical Teams** who need to understand and modify framework behavior
- **Developers** who refuse to accept "black box" solutions

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

| 🏭 **Production Frameworks** | 🔬 **TensorWeaver** |
|------------------------------|---------------------|
| ❌ Complex C++ codebase | ✅ Pure Python - fully debuggable |
| ❌ Optimized for speed only | ✅ Optimized for understanding and modification |
| ❌ "Trust us, it works" | ✅ "Here's exactly how it works" |
| ❌ Black box internals | ✅ Complete transparency and control |

### **🚀 Key Features**

- **🔍 Transparent Implementation**: Every operation is visible, debuggable, and modifiable
- **🐍 Pure Python**: No hidden C++ complexity - full control over the codebase
- **🎯 PyTorch-Compatible API**: Drop-in replacement with complete visibility
- **🛠️ Engineering Excellence**: Clean architecture designed for understanding and extension
- **🧪 Complete Functionality**: Autodiff, neural networks, optimizers, ONNX export
- **📊 Production Ready**: Export trained models to ONNX for deployment

## 🗺️ **Technical Roadmap**

### **🔧 Core Components**
1. **[Tensor Operations](milestones/01_linear_regression/)** - Fundamental tensor mechanics and operations
2. **[Linear Models](milestones/01_linear_regression/demo.ipynb)** - Basic neural network implementation
3. **Automatic Differentiation** - Gradient computation engine *(coming soon)*

### **🏗️ Advanced Architecture**
4. **[Deep Networks](milestones/03_multilayer_perceptron/)** - Multi-layer perceptron and complex architectures
5. **Optimization Algorithms** - Advanced training techniques *(coming soon)*
6. **[Model Deployment](milestones/02_onnx_export/)** - ONNX export for production systems

### **⚡ Performance & Extensions**
7. **Custom Operators** - Framework extension capabilities *(coming soon)*
8. **Performance Engineering** - Optimization techniques *(coming soon)*
9. **Hardware Acceleration** - GPU computation support *(in development)*

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

## 🎯 **Why Engineers Choose TensorWeaver**

Instead of opaque "black box" frameworks, TensorWeaver provides:
- **Full Transparency** - Every operation is readable, debuggable Python code
- **Complete Control** - Modify any component to fit your specific needs
- **PyTorch Compatibility** - Use existing knowledge and code seamlessly
- **Deep Understanding** - Know exactly what your model is doing at every step

*Join our growing community of engineers who value transparency and control.*

## 🚀 **Get Started Now**

### **📦 Installation**
```bash
# Option 1: Install from PyPI (recommended)
pip install tensorweaver

# Option 2: Install from source (for contributors)
git clone https://github.com/howl-anderson/tensorweaver.git
cd tensorweaver
uv sync --group dev
```

### **🎯 Quick Start Guide**

1. **[📂 Browse Examples](milestones/)** - Working implementations and demos
2. **[🚀 Try Online](https://mybinder.org/v2/gh/howl-anderson/tensorweaver/HEAD)** - Browser-based environment
3. **[💬 Community Forum](https://github.com/howl-anderson/tensorweaver/discussions)** - Technical discussions and support
4. **[📖 Documentation](https://tensorweaver.ai)** - Complete API reference *(expanding)*

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

## 🏢 **Professional Use Cases**

TensorWeaver excels in scenarios requiring deep understanding and control:

- **🔬 Research & Development** - Implement novel algorithms with full control
- **🐛 Debugging Complex Models** - Trace gradient flow and identify numerical issues
- **🏗️ Custom Implementations** - Build specialized layers and operators
- **📊 Production Prototyping** - Develop and export models to ONNX for deployment

*Need support for your specific use case? Open an issue or join our discussions!*

## ⭐ **Why Stars Matter**

If TensorWeaver helped you debug, understand, or build better models, please consider starring the repository! It helps other engineers discover this transparent framework.

<p align="center">
  <a href="https://github.com/howl-anderson/tensorweaver/stargazers">
    <img src="https://img.shields.io/github/stars/howl-anderson/tensorweaver?style=social" alt="GitHub stars">
  </a>
</p>

## 📄 **License**

TensorWeaver is MIT licensed. See [LICENSE](LICENSE) for details.

## 🙏 **Acknowledgments**

- Inspired by transparent implementations: **Micrograd**, **TinyFlow**, and **DeZero**
- Thanks to the PyTorch team for the elegant API design
- Grateful to all contributors and the open-source community

---

<p align="center">
  <strong>Ready for complete transparency in deep learning?</strong><br>
  <a href="https://tensorweaver.ai">🚀 Explore TensorWeaver at tensorweaver.ai</a>
</p>