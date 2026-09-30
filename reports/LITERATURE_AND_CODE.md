# 文献、开源实现与本项目适用性

检索/核对日期：2026-09-30。优先论文原文、ACL Anthology、作者项目和官方代码。定向查找CES合成受访者与PPI应用未得到足以证明“首次”的证据，因此不使用“首个/全新方法”表述。

|文献|已核对的证据与用途|能支持什么 / 不能支持什么|
|---|---|---|
|Argyle et al. (2023), *Out of One, Many: Using Language Models to Simulate Human Samples*, Political Analysis 31(3):337-351. DOI 10.1017/pan.2023.2. [原文](https://doi.org/10.1017/pan.2023.2)|人口属性条件化的“silicon samples”出发点；核对出版页面/摘要|支持设置人口学提示基线；不能保证欧元区经济预期的个体预测准确|
|Bisbee et al. (2024), *Synthetic Replacements for Human Survey Data? The Perils of Large Language Models*, Political Analysis 32(4):401-416. DOI 10.1017/pan.2024.5. [原文](https://doi.org/10.1017/pan.2024.5)|平均值相近仍可伴随方差不足、回归结构偏差和提示/时间敏感性；核对出版页面/摘要|支持同时检查个人、分布、结构与重复调用；不把其他调查的失败幅度照搬CES|
|Park et al. (2024), *Generative Agent Simulations of 1,000 People*, arXiv:2411.10109. [论文](https://arxiv.org/abs/2411.10109)|已下载原文；以1,052人的丰富访谈为信息源。85%是相对真人自身复测一致性的比例，不是85%绝对准确率|支持“更丰富个人信息可能有效”；访谈与三期结构化CES历史并不等价|
|Cao et al. (2025), *Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations*, NAACL:3141-3154. DOI 10.18653/v1/2025.naacl-long.162. [论文](https://aclanthology.org/2025.naacl-long.162/)|已下载原文；以首token概率匹配国家层面文化调查回答分布|支持概率分布目标；本次历史条件转移软标签LoRA是简化适配，不是论文完整复现|
|Suh et al. (2025), *Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions*, ACL:21147-21170. DOI 10.18653/v1/2025.acl-long.1028. [论文](https://aclanthology.org/2025.acl-long.1028/)|核对正式出版元数据和摘要；SubPOP包含3,362题、约70K子群-响应对|说明面向子群分布的微调已有成熟方向；其跨题型规模远大于本次单题LoRA，不可等同|
|Krsteski et al. (2026), *Valid Survey Simulations with Limited Human Data: The Roles of Prompting, Fine-Tuning, and Rectification*, ACL July 2026:10887-10906，DOI 10.18653/v1/2026.acl-long.498；预印本arXiv:2510.11408 (2025). [论文](https://aclanthology.org/2026.acl-long.498/)|已下载原文；比较合成与真人残差修正，作者代码可用|直接支持把“合成是否像人”与“辅助总体估计是否有效”分开；不能把论文的偏差或样本效率增益当成CES结果|
|Angelopoulos et al. (2023), *Prediction-Powered Inference*, arXiv:2301.09633. [论文](https://arxiv.org/abs/2301.09633)|核对原始摘要与作者工具库；预测加真人残差用于统计估计|支持差分估计框架；本次固定有限面板、无放回抽样与t近似区间是特定设计，不是对复杂CES调查设计的通用有效性证明|
|Paglieri et al. (2026), *Persona Generators: Generating Diverse Synthetic Personas for Arbitrary Contexts*, arXiv:2602.03545. [论文](https://arxiv.org/abs/2602.03545)|已下载v2原文；多样性/支持覆盖目标与人口分布密度目标不同|适合调查预试的极端情景覆盖；不作为提高CES代表性的主要方案。未确认可直接复用的作者公开仓库|

## 可直接采用的工程资源

|资源|核对状态|本次采用方式|
|---|---|---|
|[survey-simulations](https://github.com/skrsteski/survey-simulations)|作者仓库、MIT；commit d0926093b57b373760679c4b6d7ef01d4447c387|参考“生成+校正”的分离评价设计。本次自行实现有限面板差分估计，不安装整个Axolotl训练栈|
|[SimLLMCultureDist](https://github.com/yongcaoplus/SimLLMCultureDist)|作者仓库；commit 402ee0b350983fd665e6668d770c7a687c5669a6；GitHub未列出顶层许可证|参考首token分布目标，不复制未明确授权的实现；原始多卡/SLURM脚本不直接用于Mac|
|[ppi_py](https://github.com/aangelopoulos/ppi_py)|作者库、MIT，commit见code_manifest.json|公式与接口参考；独立有限总体实现加入无放回抽样修正，避免直接套默认超总体方差|
|[MLX-LM](https://github.com/ml-explore/mlx-lm)|官方Apple Silicon运行/微调库、MIT；使用本机已安装版本|加载现有Qwen3.5-4B 8bit，Metal运行、最后一层LoRA，保留原权重|

仓库快照清单为 `sources/papers/code_manifest.json`。前三个研究仓库只做只读核对/方法参考，不能称为本次已经完整重跑原论文。实际执行的是本项目`scripts/`内的适配实现及已安装MLX-LM。

## 数据支持链

[ECB官方数据与方法页面](https://www.ecb.europa.eu/stats/ecb_surveys/consumer_exp_survey/html/data_methodological.en.html)公开月度CSV、背景CSV、元数据和微观数据指南。稳定匿名ID允许纵向关联；数据含目标六题及国家、年龄、收入组、横截面权重。取同一当前release的2025和2026资料避免跨旧版本ID变更。由此可以直接实施滞后预测、返回面板、跨月份比较和随机隐藏标签的校正实验，无需等待新增数据采集。

This paper uses data from the ECB Consumer Expectations Survey. Source: ECB Consumer Expectations Survey.
