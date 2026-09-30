# ECB CES 消费者预期模拟

基于欧洲央行 Consumer Expectations Survey（CES），研究如何利用个人历史回答、宏观经济信息和本地语言模型预测消费者预期。

[GitHub 协作仓库](https://github.com/shimakazefeng03/ecb-ces-survey-simulation) · [关键结果](reports/KEY_RESULTS.md) · [数据获取](DATA_ACCESS.md) · [参与协作](CONTRIBUTING.md)

## 关键结果

2026 年 1、3、6 月，11 国共 660 条受访者月度记录：

- **个人历史改善预测。** DeepSeek 加入近三期个人回答后，一年期通胀预期 MAE 从 **4.06 降至 2.71** 个百分点，未来财务状况准确率从 **44.2% 提升至 69.8%**。
- **本地轻量微调有效。** Qwen3.5-4B 使用 1,024 条训练记录进行末层 LoRA，未来财务状况准确率从 **49.4% 提升至 72.9%**；实验在 Apple M3 Pro、18 GiB 内存、MLX Metal 上完成。
- **简单方法提供有力参照。** 同一财务状况任务中，近三期众数准确率为 **73.3%**；梯度提升模型使用 1,024 条和 94,850 条训练记录时分别为 **70.3%** 和 **75.5%**。

核心对比与实验设置见 [关键结果](reports/KEY_RESULTS.md)，对应数据见 [key_results.csv](reports/key_results.csv)。

## 项目结构

| 路径 | 内容 |
|---|---|
| `reports/` | 精选结果与核心指标 |
| `scripts/` | 已完成研究的数据处理、API 实验、MLX 微调与分析代码 |
| `experiments/ces_mps/` | 正在开发的 PyTorch MPS 后续实验 |
| `sources/` | 官方数据链接与文件校验记录 |

后续实验围绕宏观因子、回答变化、长历史摘要、多模型组合和本地微调展开，实验安排见 [PROTOCOL.md](experiments/ces_mps/PROTOCOL.md)。

This paper uses data from the ECB Consumer Expectations Survey. Source: ECB Consumer Expectations Survey.
