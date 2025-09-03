# Continual Learning with Prototypes for Domain Shift Adaptation

**Team: Parameter Hunters**

A comprehensive implementation of Learning with Prototypes (LWP) for continual learning across sequential CIFAR-10 datasets with varying distributions, addressing catastrophic forgetting in domain shift scenarios.

## 🎯 Problem Statement

Traditional machine learning models suffer from **catastrophic forgetting** when learning new tasks sequentially. This project tackles the challenge of training a single, fixed-capacity model on a stream of 20 CIFAR-10 subsets without losing performance on previously learned datasets.

### Dataset Structure
- **D1-D10**: Same input distribution p(x)
- **D11-D20**: Different input distributions with domain shifts
- **Constraint**: Only D1 is labeled; remaining datasets are unlabeled
- **Evaluation**: 20 held-out labeled datasets for accuracy assessment

## 🚀 Approach

### 1. Feature Extraction Benchmark
Evaluated **16 state-of-the-art architectures** across multiple paradigms:

| Architecture Type | Models | Best Performance |
|-------------------|---------|------------------|
| **Vision Transformers** | ViT-Base, Swin-Transformer | **84.68%** (ViT-Base) |
| **ConvNets** | AlexNet, VGG16, ResNet50/101 | 67.32% (ResNet101) |
| **Efficient Architectures** | EfficientNet-B0/B3, MobileNetV2 | 79.88% (EfficientNet-B0) |
| **Attention Networks** | SENet154, ConvNeXt-Base | 75.80% (ConvNeXt-Base) |

### 2. Learning with Prototypes Framework

#### Core Components:
- **LWP Hard (f1)**: Deterministic class assignments with Gaussian noise regularization
- **LWP Soft (f2-f10)**: Probabilistic assignments using softmax-based weighting
- **Mahalanobis Distance**: Advanced distance metric for robust classification
- **Prototype Smoothing**: Global mean incorporation for generalization

#### Mathematical Foundation:
```
Soft Assignment Probability:
Z_nk = exp(-||x_n - μ_k||²) / Σ_l exp(-||x_n - μ_l||²)

Weighted Mean Update:
μ_k = Σ_n Z_nk * x_n / Σ_n Z_nk

Exponential Moving Average:
μ_new = α * μ_old + (1-α) * μ_current
```

### 3. Class-Weighted Clustering Innovation

Our novel approach for enhanced continual learning:

```
Distance Weight = 1 / (1 + (Min_Mahalanobis_Distance)²)
```

**Key Benefits:**
- Prioritizes representative samples over outliers
- Maintains prototype quality during sequential updates
- Significantly improves retention across domain shifts

## 📊 Results

### Task 1: Same Distribution Learning (D1-D10)
- **Consistent Performance**: 99.96% on D1, ~87% across D2-D10
- **Minimal Forgetting**: Stable accuracy matrix with diagonal dominance
- **Strong Generalization**: Robust confusion matrix patterns

### Task 2: Domain Shift Adaptation (D11-D20)

| Method | Accuracy on D1 (f20) | Performance |
|--------|----------------------|-------------|
| **Class-Weighted Clustering** | **95.72%** | ✅ Best |
| T2PL + RandMix | 81.7% | ⚠️ Moderate |
| Generative Replay | 60.3% | ❌ Poor |

## 🛠️ Implementation

### Key Features:
- **Multi-Architecture Evaluation**: Comprehensive feature extractor comparison
- **Dual LWP Strategy**: Hard and soft assignment methodologies
- **Advanced Regularization**: Covariance matrix stabilization
- **Prototype Memory**: Exponential weighted averaging for knowledge retention
- **Domain-Aware Updates**: Adaptive learning rates for distribution shifts

### Technical Stack:
- **Deep Learning**: PyTorch, Ultralytics
- **Feature Extraction**: Pre-trained Vision Transformers
- **Evaluation**: Comprehensive accuracy matrices and confusion analysis
- **Visualization**: Performance tracking across sequential datasets

## 🎓 Key Contributions

1. **Comprehensive Architecture Analysis**: Comparison of 16 architectures for LWP-based continual learning
2. **Novel Class-Weighted Clustering**: Distance-based weighting for improved prototype updates
3. **Domain Shift Robustness**: Effective handling of distribution changes in sequential learning
4. **Catastrophic Forgetting Mitigation**: 95.72% accuracy retention across 20 sequential datasets

## 📚 Academic Context

This work addresses fundamental challenges in:
- **Continual Learning**: Sequential task learning without forgetting
- **Domain Adaptation**: Handling distribution shifts in real-world scenarios
- **Prototype-Based Learning**: Memory-efficient classification approaches
- **Vision Transformers**: Modern architectures for feature representation

### Related Work:
- Lifelong Domain Adaptation via Consolidated Internal Distribution (NeurIPS 2021)
- Deja vu: Continual Model Generalization for Unseen Domains (ICLR 2023)


## 🔗 Resources

- **Dataset**: [CIFAR-10 Continual Learning Subsets](https://tinyurl.com/cs771-mp2-data)

**Course**: CS771 - Introduction to Machine Learning (Autumn 2024)  
**Institution**: Indian Institute of Technology Kanpur  
**Submission Date**: November 22, 2024
