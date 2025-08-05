# 🧮 Numerical Methods Lab

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-green.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive collection of numerical methods implementations and experiments for scientific computing and machine learning optimization.

## 📚 Table of Contents

- [Overview](#overview)
- [Featured Projects](#featured-projects)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Key Implementations](#key-implementations)
- [Usage Examples](#usage-examples)
- [Results & Achievements](#results--achievements)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains implementations of various numerical methods with a focus on:

- **Root Finding Algorithms** (Bisection, Newton-Raphson, Secant)
- **Machine Learning Optimization** using classical numerical methods
- **Synthetic Data Generation** with SMOTE
- **Loss Function Minimization** through gradient root finding
- **Comparative Analysis** of numerical vs modern optimization techniques

## 🌟 Featured Projects

### 1. 🎯 Bisection Method for ML Loss Gradient Optimization

**Location:** `bisection method/`

A novel approach to logistic regression optimization using the classical bisection method to find roots of loss function gradients.

**Key Features:**
- ✅ Pure mathematical implementation (no external optimizers)
- ✅ Sequential optimization of weights and bias terms
- ✅ Achieves 87.5% accuracy on synthetic dataset
- ✅ Complete Bengali documentation for educational purposes
- ✅ Comprehensive visualization and analysis

**Files:**
- `gradient.ipynb` - Main implementation
- `loss_gradiant.ipynb` - Original proof of concept
- `smote_synthetic_data.ipynb` - Data generation pipeline
- `bisection_method_bangla_explanation.md` - Complete Bengali documentation

### 2. 📊 SMOTE Synthetic Data Generation

**Location:** `bisection method/smote_synthetic_data.ipynb`

Implementation of Synthetic Minority Oversampling Technique for balanced dataset creation.

**Features:**
- 🎯 Generates 1000 balanced synthetic samples
- 🎯 Perfect class distribution (50-50 split)
- 🎯 Maintains original data characteristics
- 🎯 Comprehensive visualization and validation

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or JupyterLab
```

### Clone Repository

```bash
git clone https://github.com/Sakib-Hasan3/Numerical_Method_lab.git
cd Numerical_Method_lab
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages:**
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
jupyter>=1.0.0
seaborn>=0.11.0
```

## 📁 Project Structure

```
Numerical_Method_lab/
│
├── bisection method/
│   ├── gradient.ipynb                          # Main bisection implementation
│   ├── loss_gradiant.ipynb                     # Original implementation
│   ├── smote_synthetic_data.ipynb              # Data generation
│   ├── smote_synthetic_data.csv                # Generated dataset (1000 samples)
│   ├── logistic_regression_data.csv            # Original dataset
│   └── bisection_method_bangla_explanation.md  # Bengali documentation
│
├── other_methods/                               # Additional numerical methods
│   ├── newton_raphson/
│   ├── secant_method/
│   └── interpolation/
│
├── docs/                                        # Documentation
├── results/                                     # Output files and visualizations
├── requirements.txt                             # Dependencies
└── README.md                                    # This file
```

## 🔧 Key Implementations

### Bisection Method for Root Finding

```python
def bisection_root(func, a, b, tol=1e-6, max_iter=1000):
    """
    Find root of function using bisection method
    
    Parameters:
    - func: Function to find root of
    - a, b: Initial interval [a, b]
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    
    Returns:
    - Root of the function
    """
    fa, fb = func(a), func(b)
    if fa * fb > 0:
        raise ValueError("Function does not change sign in interval")
    
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = func(c)
        if abs(fc) < tol or (b - a)/2 < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2
```

### Logistic Loss and Gradient Functions

```python
def logistic_loss(w, b=0):
    """Cross-entropy loss for logistic regression"""
    z = w * X + b
    preds = sigmoid(z)
    eps = 1e-15
    preds = np.clip(preds, eps, 1 - eps)
    return -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

def logistic_grad(w, b=0):
    """Gradient of loss with respect to weight"""
    z = w * X + b
    preds = sigmoid(z)
    return np.mean((preds - y) * X)
```

## 💻 Usage Examples

### Basic Bisection Method Usage

```python
# Load the implementation
from gradient import bisection_root, logistic_grad

# Find optimal weight
optimal_weight = bisection_root(logistic_grad, -10, 10)
print(f"Optimal weight: {optimal_weight:.6f}")
```

### Running Complete Analysis

```bash
# Open Jupyter Notebook
jupyter notebook

# Navigate to bisection method/gradient.ipynb
# Run all cells for complete analysis
```

### Generating Synthetic Data

```python
# Run SMOTE data generation
jupyter notebook "bisection method/smote_synthetic_data.ipynb"
```

## 🏆 Results & Achievements

### Performance Metrics

| Method | Dataset | Accuracy | Loss | Convergence |
|--------|---------|----------|------|-------------|
| Bisection (No Bias) | SMOTE 1000 | 85.0% | 0.395 | 15 iterations |
| Bisection (With Bias) | SMOTE 1000 | 87.5% | 0.341 | 18 iterations |
| Target Achievement | - | ~84% | - | ✅ Achieved |

### Key Achievements

- 🎯 **Successfully implemented** pure mathematical optimization for ML
- 🎯 **Achieved target accuracy** of ~84% (87.5% actual)
- 🎯 **Demonstrated equivalence** with professional libraries
- 🎯 **Created comprehensive documentation** in Bengali for education
- 🎯 **Generated synthetic datasets** with perfect class balance

### Visualizations

The project includes comprehensive visualizations:
- Data distribution comparisons (original vs synthetic vs noisy)
- Loss function landscapes
- Gradient function analysis
- Model performance comparisons
- Convergence analysis

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core programming language | 3.8+ |
| **NumPy** | Numerical computations | Latest |
| **Pandas** | Data manipulation | Latest |
| **Matplotlib** | Visualization | Latest |
| **Scikit-learn** | ML utilities and comparison | Latest |
| **Imbalanced-learn** | SMOTE implementation | Latest |
| **Jupyter** | Interactive development | Latest |

## 📖 Educational Resources

### Bengali Documentation
- **Complete explanation** in Bengali for each code cell
- **Mathematical derivations** with Bengali commentary
- **Step-by-step algorithms** explained in native language
- **Perfect for Bengali-speaking students** learning numerical methods

### Learning Outcomes
After studying this repository, you will understand:
- ✅ Classical numerical methods for root finding
- ✅ Application of numerical methods to modern ML problems
- ✅ Implementation of optimization algorithms from scratch
- ✅ Synthetic data generation techniques
- ✅ Performance analysis and visualization

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive comments (preferably in Bengali for educational value)
- Include test cases for new implementations
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/Sakib-Hasan3)
- Email: ug2102052@cse.pstu.ac.bd
- LinkedIn: [Your LinkedIn]([https://linkedin.com/in/](https://www.linkedin.com/in/mohammed-sakib-hasan-50ab08362/))

## 🙏 Acknowledgments

- **Numerical Methods Course** for providing the theoretical foundation
- **Scikit-learn community** for reference implementations
- **Jupyter Project** for the excellent development environment
- **Bengali-speaking community** for inspiration to create native language documentation

## 📊 Repository Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/Numerical_Method_lab?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/Numerical_Method_lab?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/Numerical_Method_lab)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/Numerical_Method_lab)

---

⭐ **Star this repository** if you found it helpful!

📚 **Perfect for:** Students, Researchers, Data Scientists, Machine Learning Engineers

🎯 **Level:** Intermediate to Advanced

🌍 **Language Support:** English documentation + Complete Bengali explanations
