# H3C DRL Multi-seed Refined Training

本仓库是 H3C 论文中五个 DRL baseline 的统一时序特征、多随机种子重训客户端。它包含 refined observation 契约、PPO/MAPPO 训练代码、H3C 推理适配器、只读历史模型兼容性证据、可靠 checkpoint、并行调度、评估与汇总工具。

本仓库**不包含也不启动 BOPTEST 服务端**。开始训练前，目标电脑必须已经通过 Docker 部署好兼容的 BOPTEST，并提供 12 个 worker。训练代码仅通过 HTTP API 与该服务通信。

下文从一台已安装 Docker 和 BOPTEST 的新电脑开始，按顺序执行即可完成环境配置、检查、smoke test、正式训练、断点恢复与评估。

## 1. 实验范围与 refined 契约

| Task | Algorithm | Global obs | Local obs | Action | Episode | Full cap |
|---|---|---:|---:|---:|---:|---:|
| `sz_air_ppo` | PPO | 36 | – | 1 | 672 steps | 700 epochs |
| `mz_air_ppo` | PPO | 96 | – | 5 | 672 steps | 700 epochs |
| `mz_air_mappo` | MAPPO | 96 | 36 | 5 | 672 steps | 700 epochs |
| `mz_hydro_ppo` | PPO | 51 | – | 2 | 480 steps | 700 epochs |
| `mz_hydro_mappo` | MAPPO | 51 | 36 | 2 | 480 steps | 700 epochs |

每个训练任务固定使用 4 个独立 BOPTEST 环境；确定性训练周验证时临时增加第 5 个 TestID。PPO 使用四进程 `SubprocVecEnv`，MAPPO 使用四个独立环境并发提交 HTTP step。

五个任务统一使用占用 PMV 舒适阈值 `0.5`、`ent_coef=0.02` 和 700 epoch 安全上限。epoch 100 及以前是纯 warm-up：仍更新 best 与 plateau anchor，但不累计 misses；之后每 25 epoch 检查一次，连续三次未达到 1% 显著改善才停止，因此最早停止点是 epoch 175。为避免把“增加上限”悄悄变成另一种学习率计划，原学习率衰减期限保持不变：SZ/MZ-Air 在 300 epoch 到达初始学习率的 10%，Hydronic 在 500 epoch 到达 10%，之后保持该下限直到早停或 epoch 700。

三个案例统一采用以下时序契约（一个 step 为 15 分钟）：

- 每个 zone 的温度：`T[t], T[t-1], T[t-2], T[t-3], T[t-4]`；
- 每个 zone 的物理设定值历史：`a[t-1] ... a[t-4]`；
- 全系统归一化功率：`P[t-1] ... P[t-4]`；
- 外气温度、太阳辐射、电价及逐 zone 占用预测：`x[t] ... x[t+4]`，即 0–60 分钟；
- 温度历史不足时，所有任务都重复最早可用温度；动作缺失用 25°C 初始设定，功率缺失用 0；
- MZ-Air 与 MZ-Hydronic MAPPO 均采用 `zone_then_shared` 局部列顺序，局部输入均为 36 维；
- 时间正余弦统一使用 `[-1,1]` bounds，物理设定值历史统一使用 20–30°C bounds。
- 占用预测统一保留为有效人数，不再对 MZ-Air 先二值化再按人数上界归一化；不同 testcase 的人数容量上界仍分别保留。

未来动作尚未由策略决定，因此不作为当前输入；这里的“未来四步”均指预报特征。权威定义位于 `configs/refined_observation_contracts.json`。

占用日程规则、人数容量上界、episode 长度以及 Hydronic 已批准的有人 25°C/无人 30°C 残差基准仍按 testcase 保留，这些是有记录的物理/数据差异，不属于时间索引不一致。

新 seed 均从随机初始化冷启动，不加载 `models/` 中的历史权重。旧 registry 和旧模型保持只读，可继续用于旧结果评估，但其 81/45/33 维 checkpoint **不能**用于 refined 训练续传。前两版结果保留在 `outputs_refine/` 和 `outputs_refine_uniform/`；本次修正后的 run 写入独立的 `outputs_refine_v2/`，并使用全新的 W&B project `h3c-drl-multiseed-refine-v2`。仓库启动时会同时检查 refined observation 顺序/维度、统一训练设置、早停协议和旧模型 SHA256/golden actions；不一致时拒绝训练。

## 2. 开始前检查

需要：

- Windows 10/11 PowerShell 5.1+，或 Linux/macOS 上的 PowerShell 7 / Python CLI；
- Git 和 64-bit Python 3.10；Python 可以通过命令行安装，无需 Conda；
- Docker Desktop/Engine 和 Docker Compose v2；
- 已部署的 BOPTEST `0.8.0-dev` 或 `1.0.0-dev`；
- 已 provision 以下三个 testcase：
  - `bestest_air`
  - `multizone_office_simple_air`
  - `multizone_office_simple_hydronic`
- BOPTEST worker 数量至少为 12；
- 可选：NVIDIA GPU 和可用驱动。PPO 默认 CPU；MAPPO 优先 CUDA，不可用时回退 CPU。

本仓库接受 BOPTEST `/version` 返回 `0.8.0-dev` 或 `1.0.0-dev`。参考实验使用 `0.8.0-dev`；使用 `1.0.0-dev` 时 preflight 会给出提示并把实际版本写入运行元数据，但不会阻止训练。其他未验证版本仍会被拒绝。

## 3. 检查现有 BOPTEST

以下 Docker 命令应在你自己的 **BOPTEST 安装目录**执行，不是在本训练仓库中执行。服务名以标准 BOPTEST Compose 的 `worker` 为例；若你的部署使用不同服务名，请对应替换。

```powershell
docker compose up -d --scale worker=12
docker compose ps
(docker compose ps -q worker | Measure-Object).Count
```

最后一条应输出 `12`。如果同一台机器同时跑两个训练任务，峰值会使用 10 个 TestID；少于 10 个 worker 会导致等待或 HTTP 超时，推荐严格使用 12 个。

检查 API。默认地址为 `http://127.0.0.1:8000`：

```powershell
$env:BOPTEST_URL = "http://127.0.0.1:8000"
Invoke-RestMethod "$env:BOPTEST_URL/version" | ConvertTo-Json -Depth 5
```

返回内容必须包含 `0.8.0-dev` 或 `1.0.0-dev`。若 BOPTEST 位于另一台机器，改用其可访问地址，例如：

```powershell
$env:BOPTEST_URL = "http://192.168.1.50:8000"
```

确保防火墙允许训练电脑访问该端口。不要让两台训练电脑同时使用同一个仅有 12 workers 的 BOPTEST 服务；每台训练电脑应连接自己的 BOPTEST 实例，或者为共享服务提供足够的独立容量。

## 4. 安装 Python、Clone 与创建 venv

### 4.1 在 Windows 命令行安装 Python 3.10

先检查是否已经安装：

```powershell
py -3.10 --version
```

如果命令不存在，可在 PowerShell 中通过 Windows Package Manager 安装，无需浏览器手动下载：

```powershell
winget search --id Python.Python.3.10 --exact
winget install --id Python.Python.3.10 --exact --source winget
```

安装完成后关闭并重新打开 PowerShell，再检查：

```powershell
py -3.10 --version
```

如果电脑没有 `winget`，使用 Python 官方 Windows 安装器：<https://www.python.org/downloads/windows/>。安装时启用 Python Launcher；本仓库后续用 `py -3.10` 显式选择版本，避免误用系统里的其他 Python。

如新电脑尚未安装 Git，也可以使用：

```powershell
winget install --id Git.Git --exact --source winget
```

### 4.2 Clone 私有仓库

仓库是 Private，先在新电脑登录有权限的 GitHub 账号，然后：

```powershell
git clone --depth 1 https://github.com/wlxin-nus/h3c-drl-multiseed-training.git
Set-Location h3c-drl-multiseed-training
```

`--depth 1` 只下载当前训练客户端版本，避免下载早期提交中已移除的 BOPTEST 服务端快照。训练、checkpoint、断点恢复和后续 `git pull` 均不受影响。

### 4.3 创建标准 venv 并安装锁定依赖

在仓库根目录执行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python --version
python -c "import torch, stable_baselines3, gymnasium; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

预期 Python 为 3.10。锁定依赖包括 PyTorch 2.9.0、Stable-Baselines3 2.7.0、Gymnasium 1.2.1 和 W&B 0.28.1。每次打开新的 PowerShell 都需要重新激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

若 `torch.cuda.is_available()` 为 `False`，训练仍可运行，但 MAPPO 会使用 CPU。希望使用 5090/5080 时，应先依据该机器的 NVIDIA 驱动安装与 PyTorch 2.9.0 兼容的 CUDA wheel，再重新执行上面的检查；不要在正式运行中途更换 PyTorch 环境。

如果 Windows 阻止本仓库的 `.ps1`，只对当前 PowerShell 会话临时放行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 5. W&B 设置

在线记录：

```powershell
wandb login
```

API key 只保存在本机，不要写入仓库。若当前电脑不能联网，运行命令时使用 `-WandbMode offline`；不希望启用 W&B 时使用 `-WandbMode disabled`。无论 W&B 是否可用，本地 CSV、TensorBoard、JSONL 和 checkpoint 都会继续保存。

## 6. 强制完整性检查与 preflight

仍在仓库根目录、`.venv` 已激活、BOPTEST 已运行的前提下执行：

```powershell
python scripts\make_checksums.py --verify
python scripts\self_check.py
python -m drl_multiseed.cli preflight --online --seed 1337 --endpoint $env:BOPTEST_URL
```

只有三条命令全部成功才进入训练：

- checksum 验证所有受控文件未损坏；
- self-check 验证五项模型契约、GAE、早停、容量锁和监控 Notebook；
- online preflight 验证本机运行环境、五项输入输出契约和 BOPTEST 版本。

`outputs_refine_v2/preflight.json` 是本次检查的留档。若 preflight 报版本、维度、哈希或 golden action 不一致，不要用跳过检查的方式继续训练。

## 7. 必须先做 smoke test

推荐先用将要正式运行的 seed 执行完整五任务 smoke。每个任务只提交 2 个 epoch，但会实际创建 4 个训练环境并检查 checkpoint、日志和 TestID 清理。

双任务并行电脑：

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode smoke -MaxParallel 2 -GpuSlots 1 `
  -ThreadsPerTask 4 -Resume -WandbMode online `
  -Endpoint $env:BOPTEST_URL
```

弱电脑串行：

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode smoke -MaxParallel 1 -GpuSlots 0 `
  -ThreadsPerTask 4 -Resume -WandbMode online `
  -Endpoint $env:BOPTEST_URL
```

smoke 与 full 写入不同目录；正式训练不会继承 smoke 权重。确认以下文件存在且 `run_manifest.json` 没有异常状态：

```text
outputs_refine_v2/smoke/seed1337/<task>/run_manifest.json
outputs_refine_v2/smoke/seed1337/<task>/training_metrics.csv
outputs_refine_v2/smoke/seed1337/<task>/checkpoints/latest.json
```

## 8. 正式训练

### 8.1 推荐：两项并行

适用于 5090/5080 及较强 CPU。调度器最多启动两个任务，优先配对一个 CPU PPO 和一个 CUDA MAPPO，并确保两个并发任务来自不同 BOPTEST testcase：

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 2 -GpuSlots 1 `
  -ThreadsPerTask 4 -Resume -WandbMode online `
  -Endpoint $env:BOPTEST_URL
```

如果 GPU 显存和利用率允许两个 MAPPO 同时运行，可显式使用 `-GpuSlots 2`。这只开放 GPU 调度容量；同一 building 的 PPO/MAPPO 仍不会并发，因为两个 agent 竞争同一 FMU testcase 会显著降低吞吐。

### 8.2 弱电脑：串行

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 1 -GpuSlots 0 `
  -ThreadsPerTask 4 -Resume -WandbMode online `
  -Endpoint $env:BOPTEST_URL
```

### 8.3 只运行一个任务

```powershell
.\scripts\run_task.ps1 `
  -Task mz_hydro_ppo -Seed 1337 -Mode full -Resume `
  -WandbMode online -Endpoint $env:BOPTEST_URL -Device cpu
```

可用 task 名称就是第 1 节表格中的五项。建议机器分配：

- 5090 电脑：seed 1337，`MaxParallel=2`；
- 5080 电脑：seed 2026，`MaxParallel=2`；
- 第三台电脑：可选 clean seed 42，较弱时用 `MaxParallel=1`。

每台机器使用独立 clone 和独立 `outputs_refine_v2/`，不要让多台机器写同一个网络共享目录。

### 8.4 在 full mode 中选择案例

`run_seed_suite.ps1` 通过 `-Tasks` 选择要运行的案例与算法。省略 `-Tasks` 会依次运行全部五项。

| 想训练的内容 | `-Tasks` 值 |
|---|---|
| SZ-Air PPO | `sz_air_ppo` |
| MZ-Air PPO | `mz_air_ppo` |
| MZ-Air MAPPO | `mz_air_mappo` |
| MZ-Hydronic PPO | `mz_hydro_ppo` |
| MZ-Hydronic MAPPO | `mz_hydro_mappo` |

只训练 Hydronic 的 PPO 和 MAPPO：

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 2 -GpuSlots 1 -Resume `
  -Tasks @('mz_hydro_ppo','mz_hydro_mappo')
```

只训练 MZ-Air 的 PPO 和 MAPPO：

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 2 -GpuSlots 1 -Resume `
  -Tasks @('mz_air_ppo','mz_air_mappo')
```

选择两个不同 building 的任务并行，例如 Hydronic PPO 与 MZ-Air MAPPO：

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 2 -GpuSlots 1 -Resume `
  -Tasks @('mz_hydro_ppo','mz_air_mappo')
```

即使指定 `MaxParallel=2`，同一个 building 的 PPO 与 MAPPO 也会按顺序运行，避免争用同一 FMU testcase。要真正同时运行两个任务，应选择两个不同 building。若只想精确运行一个任务，使用第 8.3 节的 `run_task.ps1 -Task ...`。

## 9. 意外停止与断点续训

`-Resume` 应始终保留。训练进程现在会自动识别 BOPTEST HTTP/Socket 暂时故障（包括 Windows 子进程管道错误 `109`、端口/缓冲区错误 `10048`、`10055`、连接重置/拒绝、超时、HTTP 429/5xx，以及 PPO 子进程因此返回的 EOF）：

1. 放弃尚未原子提交的当前 epoch；
2. 仅清理当前 run 持有的 TestID；
3. 等待 `/version` 恢复健康；
4. 使用同一 run UUID、W&B run ID 和最近完整 checkpoint 自动续训。

默认允许同一已提交 epoch 连续自动恢复 12 次，退避为 15、30、60 秒（之后封顶 60 秒），每次最长等待 BOPTEST 600 秒。只要中间有新 epoch 成功提交，连续失败计数就会清零。恢复过程记录在 `http_auto_resume.jsonl`，最新故障写入 `last_failure.json`。配置错误、NaN、checkpoint 损坏、`Ctrl+C` 等非通讯错误不会被自动吞掉。

因此短暂 HTTP 故障通常不再需要人工操作。如果自动恢复次数耗尽，或发生 Ctrl+C、重启、断电、Python 崩溃，则：

1. 确认 BOPTEST 已恢复并仍有 12 个 worker；
2. 执行 `.\.venv\Scripts\Activate.ps1` 激活相同虚拟环境；
3. 进入同一个仓库和同一个 `outputs_refine_v2/`；
4. 原样重新执行之前的训练命令。

恢复以**最近一个原子提交完成的 epoch**为边界。未完成 epoch 会重跑一次；已完成 epoch、optimizer、LR、RNG、global step、best、早停计数和 W&B run ID 都会恢复，不会从头开始。

一般无需添加新参数。若某台机器端口资源恢复较慢，可以调整等待；将尝试次数设为 `0` 可禁用自动恢复：

```powershell
.\scripts\run_task.ps1 `
  -Task mz_air_ppo -Seed 1337 -Mode full -Resume `
  -HttpAutoResumeAttempts 20 -HttpResumeBackoffSeconds 30 `
  -HttpHealthTimeoutSeconds 900 -Endpoint $env:BOPTEST_URL
```

若机器强杀后需要先查看/清理死进程遗留的 TestID 和容量租约：

```powershell
python -m drl_multiseed.cli cleanup `
  --mode full --endpoint $env:BOPTEST_URL --worker-capacity 12
```

清理逻辑只处理确认已死亡 owner 的资源，不会停止另一个仍在运行的训练任务。

把完整 run 目录复制到另一台电脑继续时，使用 `-AllowHostMigration`：

```powershell
.\scripts\run_task.ps1 `
  -Task mz_hydro_ppo -Seed 1337 -Mode full -Resume -AllowHostMigration `
  -Endpoint $env:BOPTEST_URL
```

必须复制完整的 `outputs_refine_v2/full/seed1337/mz_hydro_ppo/`，不能只复制一个模型文件。新机器上的代码 commit、配置、BOPTEST 版本和 checkpoint 哈希必须通过校验。

## 10. 早停与 best 模型

每 25 epoch 在训练窗口运行一次独立确定性 rollout，测试周在训练结束前不可访问。

- reward 越高越好，例如 `-500` 优于 `-600`；
- epoch 100 及以前不累计 misses；最早可能在 epoch 175 停止；
- 相对 plateau anchor 改善超过 1.0% 才重置 patience；
- 连续 3 次验证没有显著改善即早停，约为 75 epoch 平台期；
- 每次实际更高的确定性回报都会更新 `best`，即使提升不足 1.0%；
- 最终评估始终读取 `best`，不是 `latest`，也不是测试周选出的 checkpoint。

该规则较激进，可能错过非常缓慢的后期改善，论文报告中必须披露这一点。

### 10.1 到达预注册上限后继续训练至平台早停

所有任务在 700 epoch 达到统一安全上限。若某个 full run 到达上限但还没有满足上面的三次平台早停规则，可显式加入 `-ContinueUntilConverged`：

```powershell
.\scripts\run_task.ps1 `
  -Task sz_air_ppo -Seed 2026 -Mode full -Resume `
  -ContinueUntilConverged -Device cpu `
  -Endpoint $env:BOPTEST_URL
```

整组任务也支持相同开关：

```powershell
.\scripts\run_seed_suite.ps1 `
  -Seed 1337 -Mode full -MaxParallel 2 -GpuSlots 0 -Resume `
  -ContinueUntilConverged -Endpoint $env:BOPTEST_URL
```

该开关的严格语义是：

- 只适用于 `Mode full`；smoke 模式会拒绝运行；
- 原上限以内的训练、确定性验证点和 best 选择规则完全不变；
- 到达原上限后，以 25 epoch 为区块继续，直到确定性训练窗回报连续三次没有相对 plateau anchor 至少改善 1.0%；
- 训练回报为负数时仍是数值越高越好；是否停止只看每 25 epoch 的确定性训练窗回报，不看单 epoch 随机回报；
- 上限后的 PPO/MAPPO Actor 学习率固定为初始值的 10%，MAPPO Critic 同样固定为其初始值的 10%，不会重新升高或重新衰减；
- optimizer、RNG、global step、best、`misses`、W&B run ID 和原子 checkpoint 全部连续；意外中断后用同一条命令继续；
- 实际最高的确定性回报始终更新 best，即使提升未达到 1.0%；
- 该模式没有另设任意的 epoch 硬上限。如果回报持续显著改善，训练会继续；可随时用 Ctrl+C 安全停止，之后仍可断点恢复。

已经触发早停的 run 代表既定平台规则已经满足，加入该开关不会重新打开训练。为保证多 seed 比较公平，决定使用延长模式后应对同一案例的全部 seed 使用相同开关，并报告每个 seed 的实际停止 epoch。

从旧版本仓库产生的现有 checkpoint 首次进入延长模式时，程序只允许一次经过锁定指纹验证的“执行策略升级”，并写入 `code_upgrades.jsonl`。科学配置哈希保持不变；任何其他训练代码漂移仍会拒绝恢复。

## 11. 查看训练状态

命令行汇总：

```powershell
python -m drl_multiseed.cli status --mode full --output-root .\outputs_refine_v2
```

TensorBoard：

```powershell
tensorboard --logdir .\outputs_refine_v2\full --port 6006
```

然后打开 `http://127.0.0.1:6006`。也可以运行只读监控 Notebook：

```text
notebooks/Monitor_Training.ipynb
```

每个 task 的核心文件：

```text
outputs_refine_v2/full/seed1337/<task>/
  run_identity.json
  run_manifest.json
  preflight.json
  best.json
  training_metrics.csv
  updates.csv
  train_window_eval.csv
  http_auto_resume.jsonl  # 仅发生自动恢复后出现
  last_failure.json       # 最近故障及其 resolved 状态
  checkpoints/latest.json
  checkpoints/epoch_*.manifest.json
  tensorboard/
  wandb/
  lifecycle/
```

checkpoint 是训练进度的唯一权威来源；不要通过编辑 CSV 或 manifest 改变 epoch。

## 12. 正式评估与三 seed 汇总

训练完成后，正式评估默认串行执行，减少 BOPTEST 负载差异：

```powershell
$tasks = @(
  'sz_air_ppo',
  'mz_air_ppo',
  'mz_air_mappo',
  'mz_hydro_ppo',
  'mz_hydro_mappo'
)

foreach ($task in $tasks) {
  .\scripts\evaluate_task.ps1 `
    -Task $task -Seed 1337 -Mode full `
    -Endpoint $env:BOPTEST_URL -Device cpu
}
```

注册的测试窗口为：

- SZ-Air：day 203，672 steps；
- MZ-Air：day 199，672 steps；
- MZ-Hydronic：day 220，480 steps。

轨迹只有完整且连续时才会原子提交。评估输出包括成本、能耗、occupied zone-hours、PMV-hours、违规率、reward、动作饱和、best epoch、停止原因和实际训练步数。

在不同电脑完成 seed 后，把各自完整的 `outputs_refine_v2/full/seed<seed>/` 复制到一台汇总电脑的同一 `outputs_refine_v2/full/` 下，再执行：

```powershell
python -m drl_multiseed.cli aggregate --mode full --output-root .\outputs_refine_v2
```

汇总会生成逐 seed 结果、mean±std 和 information-fairness matrix。当前 5/7 天测试窗并不能完全回答“更长测试周期”的要求，这一限制会保留在结果报告中。

## 13. Linux/macOS 或不使用 PowerShell

`.ps1` 只是 Python CLI 的薄封装。先使用系统提供的 Python 3.10 创建并激活 venv，再执行相同的锁定依赖安装：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

环境与 BOPTEST 检查完成后，可以直接运行：

```bash
export BOPTEST_URL=http://127.0.0.1:8000
python scripts/make_checksums.py --verify
python scripts/self_check.py
python -m drl_multiseed.cli preflight --online --seed 1337 --endpoint "$BOPTEST_URL"

python -m drl_multiseed.cli suite \
  --seed 1337 --mode full --max-parallel 2 --gpu-slots 1 \
  --threads-per-task 4 --resume --wandb-mode online \
  --worker-capacity 12 --endpoint "$BOPTEST_URL" --output-root ./outputs_refine_v2
```

单任务恢复：

```bash
python -m drl_multiseed.cli train \
  --task mz_hydro_ppo --seed 1337 --mode full --resume \
  --worker-capacity 12 --endpoint "$BOPTEST_URL" --output-root ./outputs_refine_v2
```

## 14. 常见问题

### 无法连接 BOPTEST

确认 `$env:BOPTEST_URL`、Docker 端口、Windows 防火墙和 `/version`。如果 BOPTEST 不在本机，`127.0.0.1` 一定不正确。

### Preflight 报 BOPTEST 版本错误

`0.8.0-dev` 和 `1.0.0-dev` 均可运行；后者会被明确记录为非参考版本。若报告跨机器/跨 seed 汇总，应同时披露各 run 的 `boptest_version`，并先确认两个版本在相同控制器和评估窗口下的 KPI 没有不可接受的漂移。其他版本仍需先做兼容性验证。

### HTTP 超时或 TestID 长时间等待

在 BOPTEST 安装目录确认实际运行 12 个 worker。训练客户端的 `--worker-capacity 12` 只是本机调度上限，不能凭空增加 BOPTEST 服务端 worker。

### CUDA 不可用或显存不足

先检查 `nvidia-smi` 和 `torch.cuda.is_available()`。可将 `-GpuSlots 0` 或 `-Device cpu` 作为回退；不要在同一个 seed 的中途改变网络、reward 或其他科学参数。

### W&B 断网

训练会保留本地待同步日志，不会因此终止。恢复时可用 `-WandbMode offline`，之后使用 W&B 自带同步命令上传对应 run 目录。

### Run 已锁定

先确认是否有同一 task/seed 的训练进程仍在运行。不要启动第二个进程恢复同一个 run。确认旧进程已死亡后执行第 9 节 cleanup，再用 `-Resume`。

### Checksum 失败

停止训练，从 Private 仓库重新 clone 或恢复受损文件。`outputs/`、`outputs_refine/`、`outputs_refine_uniform/`、`outputs_refine_v2/`、`.env`、W&B 文件和本机凭据不属于 checksum 清单。

### `No module named h3c.runtime`

这是旧版 clone 缺少 H3C runtime 包造成的。更新仓库并重新刷新 editable install：

```powershell
git pull
.\.venv\Scripts\Activate.ps1
python -m pip install -e . --no-deps --force-reinstall
python -c "from h3c.runtime.clients import BoptestHttpClient; print('h3c.runtime OK')"
```

## 15. Repository map

- `src/drl_multiseed/`：训练器、环境、原子恢复、并行租约、评估和汇总。
- `upstream/`：H3C baseline 的 observation builder 与 policy adapter；builder 同时保持旧模型兼容和 refined 多 zone 四动作历史支持。
- `models/`：五个只读历史模型及 golden 兼容性 registry；不用于新 seed 初始化。
- `legacy/`：用于复现 optimizer/checkpoint 语义的历史实现证据。
- `configs/`：case、模型、奖励、图结构、refined observation 和实验协议配置。
- `scripts/`：PowerShell 入口、自检与 checksum 工具。
- `notebooks/Monitor_Training.ipynb`：只读三单元格监控 Notebook。
- `tests/`：contract、early-stop、GAE、capacity、checkpoint 和日志测试。
- `requirements-lock.txt`：本仓库验证过的精确 Python 依赖版本。

训练与 H3C 部署模型的输入输出兼容性见 [docs/H3C_COMPATIBILITY.md](docs/H3C_COMPATIBILITY.md)，恢复语义见 [docs/RECOVERY.md](docs/RECOVERY.md)，硬件与并行调度见 [docs/HARDWARE_AND_PARALLELISM.md](docs/HARDWARE_AND_PARALLELISM.md)。
