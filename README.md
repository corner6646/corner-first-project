# MNIST 手写数字识别 Demo（CNN + 预处理增强 + 模型缓存）

> 作品集项目：从 0 到 1 完成 **数据加载 → 训练 → 推理 → 输入鲁棒性预处理 → 可交互 Demo → 工程化落地**。

本仓库基于 MNIST 训练 CNN 模型进行手写数字识别，并使用 **Gradio** 提供网页交互 Demo：支持画板手写 / 上传图片识别，同时展示模型实际输入的 **28×28 预处理预览**，用于可解释性与调试。

---

## 目录

- [功能亮点](#功能亮点)
- [效果预览](#效果预览)
- [快速开始（Windows PowerShell）](#快速开始windows-powershell)
- [项目结构](#项目结构)
- [输入预处理（Level 2 / 路线 B）](#输入预处理level-2--路线-b)
- [模型与缓存策略](#模型与缓存策略)
- [常见问题](#常见问题)
- [Notebook](#notebook)

---

## 功能亮点

- **CNN 识别**：输出预测数字与 Top-3 概率
- **输入预处理增强（Level 2 / 路线 B）**：对用户输入进行裁剪、等比缩放、padding 居中，提升“随便写也能识别”的体验
- **预处理预览**：展示预处理后的 28×28 输入图，让你能直观看到模型“看见了什么”
- **模型缓存（训练/推理解耦）**：首次运行自动训练并保存模型，后续启动直接加载缓存模型，避免重复训练

---

## 效果预览

- 运行后打开本地地址：`http://127.0.0.1:7860`
- 页面支持：画板手写 / 上传图片
- 支持显示：预处理后的 28×28 输入预览

demo.png

---

## 快速开始（Windows PowerShell）

> 建议在仓库根目录运行，避免相对路径（如 `models/`）指向错误位置。

```bash
cd D:\Git\projects\corner-first-project

# 1) 创建并激活虚拟环境
python -m venv venv
.\venv\Scripts\Activate

# 2) 安装依赖
pip install -r requirements.txt

# 3) 启动 Demo
python app.py
```

停止服务：在终端按 `Ctrl + C`。

---

## 项目结构

```text
corner-first-project/
  app.py                 # Gradio Demo（推理 + 预处理 + 预处理预览）
  requirements.txt       # 依赖
  Digital_Recognition.ipynb
  README.md
  models/                # 运行时生成（默认不提交）
    mnist_cnn.keras
```

> `models/` 默认被 `.gitignore` 忽略，属于运行产物。

---

## 输入预处理（Level 2 / 路线 B）

目标：把“任意风格/任意位置”的用户输入，尽量规范化为接近 MNIST 的输入分布。

大致流程：
1. 灰度化
2.（可选）自动反色（使笔迹更接近 MNIST 的亮笔迹/暗背景）
3. 找到前景区域（数字有效像素）并 **裁剪**
4. **等比缩放** 到目标尺寸（例如最长边 20）
5. **padding 到 28×28 并居中**

在 Demo 中会同时输出“预处理后的 28×28 预览”，用于解释推理结果与快速定位问题输入。

---

## 模型与缓存策略

- 首次运行：若本地不存在 `models/mnist_cnn.keras`，将自动训练并保存
- 后续运行：若存在该文件，将直接加载缓存模型并启动 Demo（不会重复训练）

这使得项目更接近真实落地流程：**训练与推理解耦**。

---

## 常见问题

### 1) 为什么建议在仓库根目录运行？
因为模型路径使用相对路径（例如 `models/mnist_cnn.keras`）。不在根目录运行，可能导致程序在别的目录创建新的 `models/`，出现“找不到缓存/又训练一次”的错觉。

### 2) 终端显示 oneDNN / GPU warning 是否正常？
正常。这是 TensorFlow 的性能提示/运行环境提示，不影响 Demo 使用。

---

## Notebook

`Digital_Recognition.ipynb` 记录了基于 MNIST 的训练与分析过程，适合用于复盘实验过程与结果解释。
