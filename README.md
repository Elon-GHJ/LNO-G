# 🩸 BloodFlow-LNO
### 基于局域神经算子的血管血液动力学通用仿真框架
**A Generalizable Surrogate Model for Vascular Hemodynamics using Local Neural Operators**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg?style=flat-square)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Task](https://img.shields.io/badge/Task-Hemodynamics%20Prediction-red.svg?style=flat-square)](https://github.com/YourUsername/BloodFlow-LNO)

---

## 📖 项目简介

**BloodFlow-LNO** 实现了一个高效的**局域神经算子 (Local Neural Operator, LNO)**。它作为传统计算流体力学 (CFD) 求解器（如 FEM/FVM）的深度学习替代方案，能够实时、高精度地预测复杂血管几何中随时间演化的血流动力学场。

> [!TIP]
> **核心突破**：本项目通过学习控制方程背后的算子映射，实现了对**未知几何 (Unseen Geometries)** 的零样本泛化，彻底解决了传统深度学习模型对固定网格的依赖。

---

## ✨ 核心亮点 (Key Features)

* **🚀 零样本泛化 (Zero-Shot Generalization)**
    单一预训练模型即可直接处理未见过的血管结构（如从单一狭窄到复杂的分叉网络），无需微调。
* **🧠 混合双路径架构 (Hybrid Dual-Path)**
    创新性融合 **谱方法 (Spectral Methods)** 与 **图神经网络 (GNNs)**，完美平衡全局流场趋势与局部微小涡流细节。
* **⚡ 高效时序演化**
    支持长时间步的稳定循环推理 (Recurrent Rollout)，模拟真实心动周期内的压力与速度波动。
* **📏 分辨率无关性 (Resolution Independence)**
    学习函数空间映射，支持在低分辨率训练、高分辨率推理（Super-Resolution）。

---

## 🖼️ 结果展示 (Demo)

模型在面对**从未见过**的几何边界时，表现出了极强的鲁棒性：

| 场景类型 | 几何特征 | 血流动力学演化 (速度场 Magnitude) |
| :--- | :--- | :--- |
| **动脉狭窄** | 局部管径剧缩 | ![狭窄](https://github.com/user-attachments/assets/68146f38-8c96-4992-913d-89690f105396) |
| **血管分叉** | Y型复杂分支 | ![Y](https://github.com/user-attachments/assets/8d15d7ee-e699-49c7-baf4-dbee4567ada7) |
| **多级分歧** | 连续拓扑变化 | ![正弦](https://github.com/user-attachments/assets/1249c3fb-0f61-478a-8479-6b8edfb1cc58) |

---

## 🛠️ 方法论 (Methodology)

### 1. 任务定义
我们将时变流场的求解定义为一个自回归的时空序列预测问题。模型映射函数为 GθG_{\theta}，通过当前物理场 utu_t 预测下一时刻 ut+1u_{t+1}：

ut+1=Gθ(ut,Geometry)u_{t+1} = G_{\theta}(u_t, \text{Geometry})

### 2. 模型架构
LNO 内部采用了并行融合的 **Dual-Path** 设计：

* **谱路径 (Spectral Path)**：利用图傅里叶变换捕捉全局低频率、大尺度流场趋势。
* **物理路径 (Physical Path)**：基于局域图卷积 (Local Message Passing) 修正局部细节，消除边缘伪影。



---

## 📅 开源计划 (Roadmap)

- [x] 核心算子 LNO 模块开发
- [ ] 论文投稿与评审 (In Preparation)
- [ ] 开放预训练权重与全量数据集
- [ ] 提供基于 PyTorch Geometric 的工程化接口

**注意**：为了遵守学术规范，本项目完整源码及数据将在论文正式录用后发布。

---

## 📩 联系与支持

如果您觉得这个项目对您的研究有帮助，请点击右上角的 **Star ⭐**。这对我非常重要！

[ [项目主页](https://github.com/YourUsername/BloodFlow-LNO) ] · [ [报告问题](https://github.com/YourUsername/BloodFlow-LNO/issues) ] · [ [联系作者](mailto:your-email@example.com) ]
