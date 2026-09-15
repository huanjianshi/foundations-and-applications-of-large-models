# Assignment 02: ConvLSTM for Bouncing-Ball Prediction

本作业从零实现 ConvLSTM，并在自动生成的 32×32 弹跳小球序列上完成一步时空预测。实验包含基线、六组控制变量对照、最优组合三次复测，以及双球预测和边界噪声增强两项创新扩展。

## 文件说明

- `convlstm_bounce.py`：基线、六组对照、最优组合复测与可视化的统一程序。
- `convlstm_innovation.py`：双球多目标预测和边界噪声鲁棒性实验。
- `requirements.txt`：运行所需依赖。
- `assignment2_artifacts/results.csv`：标准实验及最优组合的完整结果。
- `assignment2_artifacts/histories.csv`：标准实验逐 epoch 指标。
- `assignment2_artifacts/innovation_results.csv`：创新实验结果。
- `assignment2_artifacts/innovation_histories.csv`：创新实验逐 epoch 指标。
- `assignment2_artifacts/*_metadata.json`：环境、数据划分和配置元数据。
- `assignment2_artifacts/*.png`：训练曲线与预测示例图。

课程指导书、报告模板和包含个人信息的实验报告不在公开仓库中发布。

## 实验设置

- 数据：脚本独立生成 2000 条训练序列和 200 条测试序列，无样本重叠。
- 输入/输出：默认使用前 4 帧预测第 5 帧；输入帧实验使用前 8 帧预测第 9 帧。
- 训练：5 epoch，批大小 64，Adam(`lr=0.001`)。
- 设备：支持 `auto`、`cpu` 和 `cuda`；已保存结果使用 NVIDIA GeForce RTX 5060 Laptop GPU。
- 可复现性：固定数据种子与模型种子，并在 GPU 计时前后同步。

## 运行方法

需要 Python 3.10 或更高版本。CUDA 可选；没有可用 GPU 时可使用 CPU。

```bash
python -m venv .venv
pip install -r requirements.txt
python convlstm_bounce.py --experiment all --device auto --output-dir assignment2_artifacts
python convlstm_innovation.py --device auto --output-dir assignment2_artifacts
```

也可以单独运行一个标准实验，例如：

```bash
python convlstm_bounce.py --experiment kernel --device auto --output-dir assignment2_artifacts
```

## 主要结果

| 实验 | 测试 MSE | 测试 MAE |
| --- | ---: | ---: |
| 基线：1 层、32 通道、3×3 卷积核 | 0.004518 | 0.013072 |
| 最优组合平均：2 层、64 通道、5×5 卷积核 | 0.002605 | 0.010170 |
| 创新一：双球预测 | 0.005805 | 0.018663 |
| 创新二：干净训练、含边界噪声测试 | 0.005905 | 0.034144 |
| 创新二：边界噪声增强训练与测试 | 0.002791 | 0.011202 |

最优组合相对基线的平均 MSE 降低 42.35%。在含边界伪回波的测试条件下，噪声增强训练使 MSE 降低 52.74%。完整精度结果和逐轮曲线见 `assignment2_artifacts/`。

## 复现说明

GPU 调度和具体硬件会造成耗时差异。相同依赖、随机种子和配置下，误差指标应与保存结果一致或非常接近。程序会验证输入输出形状、有限数值、参数量公式以及训练/测试数据不重叠。
