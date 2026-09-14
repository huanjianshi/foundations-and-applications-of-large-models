# Assignment 01: Designing a Better Neural Network

本作业使用 PyTorch 在 `sklearn.datasets.load_digits` 数据集上从零实现多层感知器，并以控制变量法比较激活函数、网络深度、优化器、学习率、L2 正则化、Dropout 和 BatchNorm。

## 文件说明

- `mlp_digits.py`：可配置的完整实验程序。
- `requirements.txt`：运行所需的 Python 依赖。
- `assignment1_artifacts/results.csv`：基线、八组实验、三次复测及失败实验的原始结果。
- `assignment1_artifacts/lr_scan.csv`：学习率扫描结果。
- `assignment1_artifacts/*.png`：十张训练损失与测试准确率曲线。

## 实验设置

- 数据：`load_digits`，1797 个样本、64 个输入特征、10 个类别。
- 预处理：输入转为 `float32` 并除以 16；标签转为 `torch.long`。
- 划分：训练集 80%、测试集 20%，分层抽样，随机种子 42。
- 训练：交叉熵损失、30 epoch、全批量训练。
- 对照原则：实验 1 至实验 7 每组仅改变一个因素。

## 运行方法

需要 Python 3.10 或更高版本。CUDA 可选；没有可用 GPU 时会使用 CPU。

```bash
python -m venv .venv
pip install -r requirements.txt
python mlp_digits.py --device auto --output-dir assignment1_artifacts
```

也可以使用 `--experiment` 单独运行，例如：

```bash
python mlp_digits.py --device auto --output-dir assignment1_artifacts --experiment exp3
```

## 主要结果

| 配置 | 测试准确率 |
| --- | ---: |
| 基线：`64 -> 32 Sigmoid -> 10`，SGD(`lr=0.1`) | 22.22% |
| 单项最佳：Adam(`lr=0.01`) | 82.22% |
| 最优组合：`128x128 + ReLU + BatchNorm + Adam(0.001)` | 97.22% |
| 失败实验：SGD(`lr=10.0`) | 36.94%（损失明显震荡） |

最优组合在固定随机种子 42 的三次独立复测中均达到 97.22%。完整数据见 [`results.csv`](assignment1_artifacts/results.csv)。

## 复现说明

GPU 调度会造成训练耗时的小幅差异；在相同依赖版本、随机种子和设备设置下，准确率应与已保存结果一致或非常接近。

