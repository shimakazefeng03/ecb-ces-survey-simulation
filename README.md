# ECB CES：项目接手与实证研究

GitHub 协作仓库：[shimakazefeng03/ecb-ces-survey-simulation](https://github.com/shimakazefeng03/ecb-ces-survey-simulation)（私有）。协作流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，数据获取见 [DATA_ACCESS.md](DATA_ACCESS.md)。仓库分享代码、报告与聚合结果；下文涉及的原始数据、逐人输出、模型缓存和外部交付物保留在本地，不随 Git 分发。

新的本地 PyTorch MPS 实验位于 [`experiments/ces_mps/`](experiments/ces_mps/)，范围见 [PROTOCOL.md](experiments/ces_mps/PROTOCOL.md)。该路径正在开发；下文“已完成”结果指既有回溯研究，不表示新的 MPS 方案已完成。

阅读入口：`reports/FINAL_REPORT.md`（可编辑）、`reports/FINAL_REPORT.html`（带图表）、`output/pdf/ECB_CES_FINAL_REPORT.pdf`（排版版）。文献及开源方案见 `reports/LITERATURE_AND_CODE.md`，原组阅读笔记见 `sources/handoff/READING_NOTES.md`。

本项目用真实ECB公开微观数据，实施个人历史提示、身份错配对照、传统预测、本机Qwen微调及真人残差校正。主结论是：历史信息有价值，LLM相对简单方法的增量不稳定；本地微调显著改善基础Qwen，但没有证明优于三期众数。结果是回溯模拟评测，不是官方人口预测或原组Sonnet复现。

原组交付物位于用户提供的Drive，未提供可核对的Git commit，不能为原项目编造版本号：

|Drive交付物|本次采用状态|
|---|---|
|`Final Deliverables/Final Report/EIB2_Final_Report.pdf`|读取摘要、目录、PDF页34–37；下载受阻，未保存原PDF或声称读完38页|
|`PROJECT_OVERVIEW.md`（EIB2_Code总览）|全文阅读，用于核对样本、题目覆盖、实验范围、基线与汇总口径|
|`EIB2_Code/`：data_processing、model_implementation、evaluation、START_HERE.ipynb、README.md、RUN_AB_TEST.md、verify_outputs.py及结果目录|只读核对目录和总览；未下载/执行整套Sonnet实验。新研究使用本项目独立脚本|
|`Biweekly Meetings/June17th`|已读CES范围、方法和工具建议，作为本次方向选择的依据|
|`Biweekly Meetings/June11th.docx`|核对为早期CH4Net/卫星甲烷讨论，排除出CES结论|
|`Resources/Papers/persona_generators_arxiv_2602.03545.pdf`|核对论文，并从作者公开页面下载原文；用于支持覆盖与分布密度的区分|

原组数值均标明为其报告自述，未纳入本次原始实验结果。

## 已完成的实验

- 2025–2026共364,819人月、74,613人、11国；连续三期历史可用者构成评测范围。
- 固定2026年1、3、6月，每月每国20人月，共660人月、650人。
- DeepSeek V4.1 Flash：六臂各660次，另132次重复调用；合计4,092条有效返回。原始日志包括完整提示、回答、usage和网络错误。
- 五类传统基线，训练最多94,850条2025记录；另有相同1,024训练记录的表格概率模型和直接转移频率对照。
- Qwen3.5-4B 8bit：基础评分、硬标签与软标签末层LoRA，分别用2025验证集选checkpoint及温度，2026测试660条；仅c3110题。
- 个体误差、分布、联合结构、留出ID、月份、权重、配对bootstrap，以及每月55/110真人标签的有限面板校正评测。

## 目录与结果

|路径|内容|
|---|---|
|`data/raw/`|原始ECB CSV；`sources/source_checksums.json`记录下载文件哈希|
|`sources/ecb/`|方法说明、题目定义、元数据、下载链接|
|`data/processed/`|滞后特征、训练/验证/测试输入、冻结前缀缓存|
|`results/deepseek_raw.jsonl`|正式API日志，4,092条成功、41条错误记录|
|`results/deepseek_invalid_wording_excluded.jsonl`|早期题干错误的830条返回，所有正式指标均排除|
|`results/metrics.csv`|各题、各方法、月份、见过/未见训练ID、加权/不加权指标|
|`results/paired_bootstrap.csv`|DeepSeek本人历史与各对照的配对区间|
|`results/full_frame_metrics.csv`|传统方法在34,619条完整留存评测范围上的指标|
|`results/correction.csv`|每月、真人标签数、预测器的误差、覆盖与方差比|
|`results/qwen/{base,hard,soft}/`|正式配置、原始logit、adapter、训练与选择记录|
|`results/qwen_metrics.csv`|本地模型概率与准确率指标|
|`results/qwen_probability_controls.csv`|同1,024训练记录的概率对照|
|`results/qwen_paired_bootstrap.csv`、`qwen_slices.csv`|本地硬标签模型的配对比较及分组结果|
|`results/spending_common_cases.csv`|支出题共同非弃答案例，避免跨不同样本直接比较|
|`results/repeatability.csv`、`structure.csv`、`weight_sensitivity.csv`|重复调用、题间结构、权重截尾敏感性|
|`review-stage/`|DeepSeek审查原文及逐项回应；模型评语经过人工逻辑/代码核对|
|`figures/`|PNG与可导出PDF图表|
|`reports/ARTIFACT_MANIFEST.jsonl`|报告版本记录；无时间戳文件为最新版本|

`results/qwen/hard_uncached_warmup`及`hard_seed_setup_warmup`是废弃预试，不进入正式指标。正式硬标签与软标签使用相同显式随机种子20260930。训练只有一个种子；样本bootstrap不能衡量重新训练的随机性。

## 环境

主分析Python 3.12虚拟环境位于`.venv`；关键版本：pandas 2.2.3、numpy 2.3.5、scipy 1.18.1、scikit-learn 1.9.1、matplotlib 3.11.2、ReportLab 4.4.9。完整已安装包快照为 `results/environment_main.txt`，其中包含继承的文档运行时包，并非每个包都必要。

Qwen使用既有 `/Users/fengyang/Desktop/research/new/.venv-qwen35-mlx/bin/python`，MLX 0.32.2、MLX-LM 0.31.3；完整快照为 `results/environment_qwen.txt`。设备Apple M3 Pro、18 GiB内存，后端MLX Metal，**不是PyTorch MPS**。模型为 `/Users/fengyang/Desktop/research/new/experiments/qwen35_mlx/checkpoint`，保留原权重。模型路径在`run_qwen.py`中明确配置。

PDF使用macOS系统STHeiti字体；其他系统需在`build_report.py`中指定支持中文的TrueType字体。图表标签使用英文，正文为中文。

## 不调用API、不重新训练的离线重算

在项目根目录执行：

```sh
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/supplementary.py
.venv/bin/python scripts/analyze_qwen.py
.venv/bin/python scripts/qwen_controls.py
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/build_report.py
```

`qwen_controls.py`仅重新拟合1,024条数据的小型表格模型，不训练LLM。`build_report.py`从结果表生成Markdown、HTML、PDF及带时间戳副本。API价格估算随已保存审查usage更新；不是账单。

## 从原始数据重跑

原始三份CSV已随项目保留；官方链接见 `sources/ecb/download_links.json`。若重新下载，先检查release与哈希；ECB更新后的CSV不保证相同结果。`prepare_data.py`现在直接读取CSV重建初始panel，其内容已与原实验输入核对。

```sh
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/run_baselines.py
.venv/bin/python scripts/run_deepseek.py
.venv/bin/python scripts/run_deepseek.py --repeat
.venv/bin/python scripts/run_deepseek.py --ablation
.venv/bin/python scripts/prepare_qwen.py
/Users/fengyang/Desktop/research/new/.venv-qwen35-mlx/bin/python scripts/run_qwen.py --mode base
/Users/fengyang/Desktop/research/new/.venv-qwen35-mlx/bin/python scripts/run_qwen.py --mode hard
/Users/fengyang/Desktop/research/new/.venv-qwen35-mlx/bin/python scripts/run_qwen.py --mode soft
```

然后运行上述离线重算命令。DeepSeek凭据仅由进程从macOS Keychain服务`deepseek-api-key`读取，脚本不保存或打印凭据。模型别名`deepseek-flash`已在本次运行验证为V4.1 Flash；未来别名可能更新。提示调用temperature=.7，独立输出六题JSON。

API与Qwen评分会跳过已有输出中的已完成案例；原地再次执行不是新的独立重复实验。若更换数据、模型或提示，请在新的工作副本运行，使用新的输出目录与冻结前缀缓存，保留本次原始证据。缓存按输入token索引且依赖本次冻结模型，不能跨模型版本混用。`data/processed/qwen_prefix`约2.8GB，可重新计算，适配器和最终logit已另行保存。

## 解释约定

- 一年/三年题是主观年通胀预期；三年题不是累计三年通胀。
- c7010含信用、储蓄和亲友借款；早期错误题干的整批输出已排除。
- 主结果为国家均衡样本、不加权。加权敏感性使用ECB权重除以抽样概率，仍不能修复留存选择。
- 配对区间按国家分层、受访者聚类；均为探索性95%区间，无多重比较校正。
- 差分校正的目标是每月固定220人月测试面板；无放回抽真人标签。不是对整个欧元区的代表性证明。
- 每次校正重复实验共享抽样标签；模型在抽标签前固定。均值无偏不意味着小样本t区间精确覆盖，重尾通胀尤其如此。
- 未见训练ID只意味着未进入本次训练记录，不意味着基础模型预训练未见。当前数据版本和预训练污染限制已在报告明确。

This paper uses data from the ECB Consumer Expectations Survey. Source: ECB Consumer Expectations Survey.
