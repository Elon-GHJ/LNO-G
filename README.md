# BloodFlow-LNO: 基于局域神经算子的血管血液动力学通用仿真框架

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Task](https://img.shields.io/badge/Task-Hemodynamics%20Prediction-red.svg)](https://github.com/YourUsername/BloodFlow-LNO)

> **A Generalizable Surrogate Model for Vascular Hemodynamics using Local Neural Operators.**
>
> 本项目实现了一个高效的局域神经算子 (LNO)，作为传统 CFD 求解器 (FEM/FVM) 的替代方案，用于实时预测血管内复杂、时变的血流动力学流场动力学演化。

---

## 🚀 核心亮点 (Key Features)

本项目不仅是一个简单的流场预测模型，其核心突破在于**几何泛化性**与**混合架构设计**：

* **零样本泛化 (Zero-Shot Generalization)**: 训练好的单一模型可直接适应**未曾见过 (Unseen) 的血管几何外形**（如从动脉狭窄泛化到复杂的血管分支结构），无需重新训练或微调。
* **混合双路径架构 (Hybrid Dual-Path)**: 创新性地融合了**谱方法 (Spectral Methods)** 与**图神经网络 (GNNs)**，兼顾全局血管树的流场趋势与局部斑块、狭窄处的细微血流动力学特征。
* **高效时序预测**: 将血流动力学问题建模为 2D/3D 时序预测任务，支持长时间的稳定循环推理 (Recurrent Rollout)。

## 💡 为什么选择神经算子 (Why Neural Operators)?

与传统的深度学习方法（如纯 CNN 或 U-Net）不同，神经算子学习的是函数空间之间的映射，而非固定网格上的数值映射。这赋予了模型独特的优势：

1.  **边界条件泛化 (Boundary Generalization)**: 模型学习到了控制方程 (Navier-Stokes) 背后适用于血管流动的算子规律。因此，仅需训练一次，即可直接迁移应用到具有不同入口流量/压力边界条件的血管段上。
2.  **分辨率无关性 (Resolution Independence)**: 模型在低分辨率网格上训练后，可以直接在任意高分辨率网格上进行推理 (Zero-Shot Super-Resolution)，而无需重新训练。

## 🖼️ 结果展示 (Demo)

与传统 CFD 需要为每种血管几何单独建模求解不同，以下结果均由同一个预训练 LNO 模型在不同几何边界条件下直接推理得到。

| 几何类型 (Vessel Geometry Type) | 血流动力学演化 (速度场 Magnitude) |
| :--- | :--- |
| **动脉狭窄 (Arterial Stenosis)**<br>(Unseen Geometry) | [插入 stenosis.gif] |
| **血管分支 (Vascular Bifurcation)**<br>(Unseen Geometry) | [插入 bifurcation.gif] |
| **动脉瘤 (Aneurysm)**<br>(Unseen Geometry) | [插入 aneurysm.gif] |

## 🛠️ 方法论 (Methodology)

### 1. 任务定义
我们将时变流场的求解定义为一个自回归的时空序列预测问题。模型映射函数为 GθG_{\theta}，输入当前时刻物理场 utu_t，预测下一时刻物理场 ut+1u_{t+1}:
ut+1=Gθ(ut,Geometry)u_{t+1} = G_{\theta}(u_t, \text{Geometry})
通过循环调用 (Recurrently Calling)，模型可模拟长时间范围内的流体动态演化。

### 2. 模型架构: 混合双路径 (Hybrid Dual-Path)
为了解决单一网络难以同时捕捉血管几何和流场的多尺度特征问题，LNO 内部采用了并行融合架构：
* **谱路径 (Spectral Path)**: 捕捉全局低频趋势。利用**图傅里叶变换**等将信号映射到谱空间。
* **物理路径 (Physical Path)**: 修正局部细节。基于**局域图卷积**捕捉局部细节和修正谱方法带来的伪影。

---

## 📩 代码与开源 (Code & Open Source)

论文正在撰写与投稿中 (Paper in preparation)。为了遵守学术规范，本项目的完整源代码、预训练权重及数据集将在论文正式发表/录用后第一时间在本仓库开源。

如果您对本项目感兴趣，欢迎 Star ⭐ 本仓库以获取最新动态。
