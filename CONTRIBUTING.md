# 协作说明

项目仓库：https://github.com/shimakazefeng03/ecb-ces-survey-simulation

本仓库采用私有可见性。仓库所有者在 GitHub 的 Settings → Collaborators 中邀请协作者；受邀者接受后即可访问。不要通过提交文件共享访问令牌或 API 密钥。

## 开始协作

```sh
git clone https://github.com/shimakazefeng03/ecb-ces-survey-simulation.git
cd ecb-ces-survey-simulation
git switch -c feat/your-change
```

完成修改后提交到自己的分支并发起 Pull Request。PR 写清问题、修改内容、验证方式，以及是否重新运行实验；不要把环境检查或脚本语法通过描述成实验完成。

## 两套实验路径

- `scripts/`、`reports/FINAL_REPORT.md` 和根目录 `results/` 的聚合表记录已有回溯研究。旧 Qwen 实验使用 MLX Metal，并且部分脚本保留原机器的模型绝对路径，需要按运行环境调整。
- `experiments/ces_mps/` 是正在开发的 PyTorch MPS 后续实验。范围和证据约定见 `PROTOCOL.md`，环境声明见 `env-spec.json`。这部分源码的提交不表示全部实验已经完成。
- 新实验使用独立输出目录，保留已有结果和原始权重。不要在他人的正在运行目录中切换分支、重装环境或覆盖缓存。

## 数据和大文件

原始数据、逐人预测、API 原始日志、模型权重、checkpoint 和环境目录不随 Git 提交。数据获取说明见 `DATA_ACCESS.md`。仓库保留聚合指标、图表、报告及官方数据链接；现有报告中的本地日志路径是证据位置，不保证这些文件随仓库提供。

新增聚合结果前，检查表中没有受访者 ID、逐人回答或凭据。`.gitignore` 不是数据审查的替代品。不要用 `git add -f` 提交被排除的数据与权重。

## 实验记录

记录代码 commit、输入数据 release/hash、模型 ID/revision、后端、种子、参数、训练/验证划分及实际运行命令。先固定评分程序与划分，再比较模型。历史回溯结果和真正未见的未来波次结果分别报告；MPS 与 MLX 的性能记录分别注明。
