# 数据获取与本地复现

本仓库不再分发 CES 逐人微观数据或模型权重。各协作者从官方渠道获取数据，并遵守官方用途说明。

## CES

- 官方入口：https://www.ecb.europa.eu/stats/ecb_surveys/consumer_exp_survey/html/data_methodological.en.html
- 已记录的下载链接：`sources/ecb/download_links.json`。
- 已有研究使用的文件校验记录：`sources/source_checksums.json`。官方文件后续可能修订；重新下载得到的 hash 不一定与原实验相同，不应声称是原快照的完全复现。

旧版研究预期 `data/raw/` 下有以下文件：

```text
ecb.CES_data_background.en.csv
ecb.CES_data_2025_monthly.en.csv
ecb.CES_data_2026_monthly.en.csv
```

下载这些官方文件并保留文件名后，按根目录 README 的顺序准备数据和运行。该步骤可能发起大量模型调用；查看各脚本和环境要求后再执行。

MPS 后续实验的 `experiments/ces_mps/acquire_ces.py` 下载更多年份到独立目录，并优先复用本地已有的年度数据；当前版本仍要求上述背景文件已放在 `data/raw/`。宏观数据获取代码是 `experiments/ces_mps/acquire_macro.py`。这两个脚本处于后续实验开发流程内，状态与约定以 `experiments/ces_mps/PROTOCOL.md` 为准。

## 模型与凭据

模型按其官方模型卡和许可自行下载，路径保留在本地。MPS 实验的模型和环境声明见 `experiments/ces_mps/env-spec.json`。

旧 API 实验通过 macOS Keychain 的 `deepseek-api-key` 服务读取凭据，不包含可共享密钥；本地 MPS 后续实验与该 API 路径分开。不要把密钥写入 Git、issue 或报告。

Data source: ECB Consumer Expectations Survey.
