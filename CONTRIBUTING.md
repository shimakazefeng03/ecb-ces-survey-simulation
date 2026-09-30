# 参与协作

[项目仓库](https://github.com/shimakazefeng03/ecb-ces-survey-simulation)为私有仓库。所有者通过 GitHub **Settings → Collaborators** 邀请成员，成员接受邀请后即可访问。

## 开始

```sh
git clone https://github.com/shimakazefeng03/ecb-ces-survey-simulation.git
cd ecb-ces-survey-simulation
git switch -c feat/your-change
```

在自己的工作副本中修改，通过 Pull Request 合并。PR 简要写明改动、验证方式和实验结果。

## 代码与结果

- `scripts/`：已完成研究，Qwen 使用 MLX Metal。运行前按本机环境设置模型路径。
- `experiments/ces_mps/`：PyTorch MPS 后续实验，安排见 [PROTOCOL.md](experiments/ces_mps/PROTOCOL.md)。
- `reports/KEY_RESULTS.md` 与 `reports/key_results.csv`：协作展示的精选结果。
- 完整结果、图表、原始数据、模型权重和运行日志保存在本地。数据入口见 [DATA_ACCESS.md](DATA_ACCESS.md)，API 凭据使用本机凭据管理器。

## 实验记录

每次实验使用独立输出目录，记录代码 commit、数据版本与哈希、模型版本、计算后端、随机种子、数据划分及运行命令。结果表注明任务、样本量、训练数据量和指标，方便协作者核对与复现。
