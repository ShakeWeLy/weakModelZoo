# 服务器训练说明（project2_2611）

本目录是在服务器上训练和测试 SGC-YOLOv5 / YOLOv5 的脚本。所有命令都在**仓库根目录**执行，只使用仓库内的相对路径，不依赖本机绝对路径。

| 文件 | 作用 |
|---|---|
| `setup_env.sh` | 创建 conda 环境（默认 `py312`，Python 3.12.12），安装 torch 2.6.0（CUDA 轮子）和固定版本依赖 |
| `requirements-server.txt` | 服务器依赖，版本与本机 base 环境一致 |
| `check_server.py` | 开训前自检：CUDA、依赖、数据数量、模型前向；`--smoke` 额外跑一次 2 epoch 冒烟 |
| `manifest.csv` | 实验清单，`enabled=1` 的行才会运行 |
| `run_queue.py` / `run_queue.sh` | 训练队列：按 GPU 并行，训练 + 测试，已完成的自动跳过 |
| `collect_results.sh` | 按 run 名前缀打包结果（默认不含 `.pth`），用于拷回本机 |
| `common.py` | 共用的路径和数据检查 |

## 1. 拉取代码

```bash
git clone https://github.com/ShakeWeLy/weakModelZoo.git
cd weakModelZoo
git checkout main && git pull
```

## 2. 准备数据（不在 git 中）

`data/` 被 `.gitignore` 忽略，需要单独拷贝完整的 Det_Fly 数据集，目录结构如下：

```
data/Det_Fly/
  data.yaml            # nc: 1, names: [drone]
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt
```

期望数量：train 7000 / val 1500 / test 1501。数量不符时队列会拒绝启动。

两种放法：
- 直接放到 `<仓库>/data/Det_Fly`；
- 或先 `export DET_FLY_DIR=/path/to/Det_Fly`，再执行第 3 步，脚本会建立软链。

## 3. 安装环境

```bash
bash experiments/detection/project2_2611/server/setup_env.sh
conda activate py312
```

- 默认 `TORCH_CUDA=cu124`。驱动较旧时改用 `TORCH_CUDA=cu118` 或 `cu126`（PyTorch 2.6 提供的轮子）。
- 之后每次登录都要先 `conda activate py312`。

## 4. 自检

```bash
python experiments/detection/project2_2611/server/check_server.py
python experiments/detection/project2_2611/server/check_server.py --smoke
```

- 第一条：环境、GPU、数据数量，以及两个模型的前向。
- 第二条额外跑一次 2 epoch 的训练加验证集评估（测试整条链路），结束后删除临时 run。
- 全部通过后再开训。

## 5. 编辑实验清单

`manifest.csv` 字段说明：

| 字段 | 说明 |
|---|---|
| `name` | 实验名。run 目录名为 `<前缀>_<name>`，前缀默认是当天日期 |
| `model` | `configs/models/<model>.toml` 的名称 |
| `seed` | 写入 `train.seed` |
| `epochs` | 留空表示使用配置默认值 |
| `enabled` | 1 运行，0 跳过 |
| `overrides` | 用分号分隔的 `KEY=VALUE`，等价于 `train.py --set`，例如 `train.amp=false;model.params.variant=s`。不要用逗号 |

以 `#` 开头的行会被忽略。

当前清单只包含**噪声基线**：本模型与 YOLOv5s 各新增 2 个种子（43、44，40 epoch）。加上本机已有的 seed=42 结果，每个配置共 3 个种子。

候选改动（去掉 DFC、残差改恒等、CBAM 去留等）需要先在配置中实现，再往清单里加一行。

## 6. 启动训练队列

```bash
mkdir -p experiments/detection/project2_2611/server/logs
nohup bash experiments/detection/project2_2611/server/run_queue.sh --gpus 0 \
  > experiments/detection/project2_2611/server/logs/queue.out 2>&1 &
tail -f experiments/detection/project2_2611/server/logs/queue.out
```

常用参数：
- `--dry-run`：只打印计划和命令，不启动。
- `--gpus 0,1`：每张卡同时跑一个任务。
- `--num-workers 8`：DataLoader 进程数，默认 12（覆盖 `global.toml` 的 24）。
- `--prefix 2026-11-17`：指定 run 名称前缀。
- `--allow-mismatch`：数据数量不符时仍然启动，一般不要用。

队列行为：
- 已完成（`predictions/test/metrics.json` 存在）→ 跳过。
- 训练已结束但缺测试 → 只补跑 `analyze.py`。
- 目录存在但未完成（running / failed / interrupted）→ 跳过并提示。确认后手动删除该目录再重跑。
- 每个任务的 stdout 在 `logs/<run>.stdout.log`，汇总在 `logs/queue_summary.csv`。

## 7. 回收结果

```bash
bash experiments/detection/project2_2611/server/collect_results.sh 2026-11-17
# 连同权重一起打包：
bash experiments/detection/project2_2611/server/collect_results.sh 2026-11-17 --with-ckpt
```

输出在 `server/results/results_<前缀>_<时间>.tar.gz`。用 scp / rsync 拷回本机，解压到仓库同一位置即可。

## 8. 注意事项

- 队列始终带 `--no-overwrite`。直接运行 `train.py -n <已有名称>` 会删除已有目录，请避免。
- 同一 seed 在 GPU 上也不完全可复现：单次结果的波动可达 0.25–1.8 点（见实验报告表 7）。比较配置时需要多个种子。
- 长训练的稳定性尚未解决：c3_sgdown_gbot_dw5 的 200 epoch 在 E45–48 验证 mAP50 骤降并早停；m 规模模型出现过 NaN。超过 40–100 epoch 的结果先不要下结论。
- `logs/` 与 `results/` 不入库（已加入 `.gitignore`）。
- 本目录外的分析脚本（如 `diagnose_*`、`size_ap_eval.py`、`bench_latency.py`）依赖本机的运行目录，服务器上不需要。
