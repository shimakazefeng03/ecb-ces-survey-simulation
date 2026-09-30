# 数据获取与本地复现

## CES 数据

从 [ECB 官方数据页面](https://www.ecb.europa.eu/stats/ecb_surveys/consumer_exp_survey/html/data_methodological.en.html)获取数据。项目保留了 [下载链接](sources/ecb/download_links.json) 和 [研究所用文件的校验记录](sources/source_checksums.json)，复现时记录所用数据版本及哈希。

已完成研究使用以下文件，放入 `data/raw/`：

```text
ecb.CES_data_background.en.csv
ecb.CES_data_2025_monthly.en.csv
ecb.CES_data_2026_monthly.en.csv
```

## 运行入口

主分析使用 Python 3.12，主要依赖为 pandas、NumPy、SciPy 和 scikit-learn。准备数据并运行传统基线：

```sh
python scripts/prepare_data.py
python scripts/run_baselines.py
```

API 实验入口为 `scripts/run_deepseek.py`；本地 Qwen 实验先运行 `scripts/prepare_qwen.py`，再运行 `scripts/run_qwen.py`，为 `--mode` 选择 `base`、`hard` 或 `soft`。Qwen 使用 MLX 0.32.2、MLX-LM 0.31.3 和本地 Qwen3.5-4B 8bit 权重，模型路径在 `run_qwen.py` 中配置。DeepSeek 凭据通过 macOS Keychain 服务 `deepseek-api-key` 读取。

分析入口为 `scripts/analyze.py` 和 `scripts/analyze_qwen.py`。生成的完整指标与日志保存在本地，精选结果整理至 `reports/`。

## MPS 后续实验

环境与模型配置见 [env-spec.json](experiments/ces_mps/env-spec.json)，实验安排见 [PROTOCOL.md](experiments/ces_mps/PROTOCOL.md)。

- `experiments/ces_mps/acquire_ces.py`：获取 2020–2026 年月度数据，优先复用已有年度文件；背景文件从 `data/raw/` 读取。
- `experiments/ces_mps/acquire_macro.py`：获取宏观经济因子。
- `experiments/ces_mps/download_models.py`：获取本地模型。

Data source: ECB Consumer Expectations Survey.
