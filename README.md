# EdgeSR：边缘感知图像超分辨率模型

**计算机视觉课程项目**

基于 EDSR 架构改进的边缘感知超分辨率模型，包含 EARB（Edge-Aware Residual Block）和 LCAP（Lightweight Channel Attention Pruning）两个核心模块。

## 项目内容

- 实现 EDSR Baseline 作为对比基准
- 设计并实现 EdgeSR，包含 EARB 和 LCAP 两个模块
- 在 Set5、Set14、BSD100 上测试，与 Baseline 对比
- 通过门控分析、边缘可视化、差值图等方法分析模型行为
- 探索盲 SR（退化增强、真实数据、二阶退化等方向）

## 实验结果

| 模型（×2）| 参数量 | Set5 | Set14 | BSD100 |
|---------|--------|------|-------|--------|
| Bicubic | — | 33.66 / 0.930 | 30.23 / 0.869 | 29.56 / 0.843 |
| EDSR Baseline | 1.44M | 35.86 / 0.951 | 31.64 / 0.906 | 30.86 / 0.902 |
| EdgeSR（本文）| 1.44M | 35.85 / 0.951 | 31.52 / 0.904 | 30.84 / 0.902 |

详细报告见 `docs/report.md`。

## 快速开始

```bash
# 训练 EDSR Baseline
python -m src.train --config configs/baseline.yaml --model baseline

# 训练 EdgeSR 标准版
python -m src.train --config configs/edgesr_standard.yaml --model edgesr

# 测试
python -m src.test --config configs/edgesr_standard.yaml --model edgesr --checkpoint checkpoints/edgesr_standard_best.pt

# 使用自集成推理
python -m src.test --config configs/edgesr_standard.yaml --model edgesr --checkpoint checkpoints/edgesr_standard_best.pt --self-ensemble

# 启动 Gradio 网页推理界面
python src/gradio_app.py --config configs/edgesr_standard.yaml --model edgesr --checkpoint checkpoints/edgesr_standard_best.pt --port 7860
```

## 模型结构

- `configs/` — 每个实验独立的配置文件
- `src/models/` — 模型定义（EDSR Baseline、EdgeSR、EARB、LCAP 等）
- `src/data/` — 数据加载与退化流水线
- `src/scripts/` — 门控分析与边缘可视化脚本
- `logs/` — 训练日志
- `checkpoints/` — 预训练权重

## 实验配置

| 配置文件 | 实验内容 |
|---------|---------|
| `baseline.yaml` | EDSR Baseline |
| `edgesr_standard.yaml` | EdgeSR 标准训练 |
| `edgesr_degrad.yaml` | 退化增强训练 |
| `edgesr_realsr.yaml` | RealSR 数据训练 |
| `edgesr_drealsr.yaml` | DRealSR 数据训练 |
| `edgesr_drealsr_ft.yaml` | DRealSR 微调 |
| `edgesr_nolcap.yaml` | 消融实验（无 LCAP）|
| `edgesr_pruned.yaml` | 剪枝微调 |
| `edgesr_ssim.yaml` | L1+SSIM 损失 |
