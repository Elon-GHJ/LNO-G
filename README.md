# Vehicle-LNO: 基于局域神经算子的车辆空气动力学通用仿真框架

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Task](https://img.shields.io/badge/Task-2D_Unsteady_Flow_Prediction-red)]()

> **A Generalizable Surrogate Model for Vehicle Aerodynamics using Local Neural Operators.**
>
> 本项目实现了一个高效的局域神经算子 (LNO)，作为传统 CFD 求解器 (FEM) 的替代方案，用于实时预测瞬态可压缩流场的动态演化。

---

## 🚀 核心亮点 (Key Features)

本项目不仅仅是一个简单的流场预测模型，其核心突破在于**几何泛化性**与**混合架构设计**：

* **零样本泛化 (Zero-Shot Generalization)**: 训练好的单一模型可直接适应**未曾见过 (Unseen)** 的车辆几何外形（如从轿车泛化到卡车），无需重新训练或微调。
* **混合双路径架构 (Hybrid Dual-Path)**: 创新性地融合了**谱方法 (Spectral Methods)** 与**卷积神经网络 (CNN)**，兼顾全局流场趋势与局部激波细节。
* **高效时序预测**: 将 CFD 问题建模为 **2D 时间序列预测** 任务，支持长时间的稳定循环推理 (Recurrent Rollout)。

---

## 💡 为什么选择神经算子 (Why Neural Operators)?

与传统的深度学习方法（如纯 CNN 或 U-Net）不同，本课题采用的**神经算子 (Neural Operator)** 学习的是函数空间之间的映射，而非固定网格上的数值映射。这赋予了模型独特的优势：

1.  **边界条件泛化 (Boundary Generalization)**: 模型学习到了控制方程（Navier-Stokes）背后的算子规律。因此，**仅需训练一次**，即可直接迁移应用到具有**不同输入形状**和**不同边界条件**的仿真区域上。
2.  **分辨率无关性 (Resolution Independence)**: 模型在低分辨率网格上训练后，可以直接在任意高分辨率网格上进行推理 (Zero-Shot Super-Resolution)，而无需重新训练。

---

## 🖼️ 结果展示 (Demo)

与传统 CFD 需要为每种车型单独建模求解不同，以下结果均由**同一个预训练 LNO 模型**在不同几何边界条件下直接推理得到。

| **几何类型** | **流场演化 (速度场 magnitude)** |
| :---: | :---: |
| **轿车 (Sedan)**<br>*(Unseen Geometry)* | ![跑车流场](轿车.gif) |
| **SUV**<br>*(Unseen Geometry)* | ![SUV流场](SUV.gif) |
| **卡车 (Truck)**<br>*(Unseen Geometry)* | ![卡车流场](truck.gif) |

---

## 🧠 方法论 (Methodology)

### 1. 任务定义
我们将非定常流场的求解定义为一个自回归的**时空序列预测问题**。
模型映射函数为 <img src="https://latex.codecogs.com/svg.latex?\mathcal{G}_\theta" title="\mathcal{G}_\theta" />，输入当前时刻物理场 <img src="https://latex.codecogs.com/svg.latex?u_t" title="u_t" />，预测下一时刻物理场 <img src="https://latex.codecogs.com/svg.latex?u_{t+1}" title="u_{t+1}" />：

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?u_{t+1}%20=%20\mathcal{G}_\theta(u_t,%20\text{Geometry})" alt="Equation" />
</div>

通过循环调用 (Recurrently Calling)，模型可模拟长时间范围内的流体动态演化。

### 2. 模型架构：混合双路径 (Hybrid Dual-Path)
为了解决单一网络难以同时捕捉流场多尺度特征的问题，LNO 内部采用了并行融合架构：

* **🌊 谱路径 (Spectral Path) - 主导路径**
    * **原理**: 利用 **勒让德谱变换 (Legendre Transform)** 将输入映射到模态空间。
    * **作用**: 高效捕捉流场的**全局低频趋势**和能量的大尺度传递。
    * **优势**: 具有全局感受野，计算效率高。

* **⚡ 物理路径 (Physical Path) - 修正路径**
    * **原理**: 基于 **局部 CNN (Local CNN)** 直接在物理网格上运行。
    * **作用**: 捕捉**高频局部细节**，专门负责修正谱方法在激波或复杂边界处产生的吉布斯现象 (Gibbs Phenomenon/数值振荡)。
    * **优势**: 对局部梯度敏感，增强了模型的鲁棒性。

---

## 🚧 代码与开源 (Code & Open Source)

**论文正在撰写与投稿中 (Paper in preparation).**

为了遵守学术规范，本项目的完整源代码、预训练权重及数据集将在**论文正式发表/录用后**第一时间在本仓库开源。

如果您对本项目感兴趣，欢迎 **Star** ⭐ 本仓库以获取最新动态。

---
