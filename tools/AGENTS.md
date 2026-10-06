# tools/hpc.py — Agent 用 HPC 调用工具（乌镇 SLURM 集群）

> 本文件面向 AI agent。人类简报在 hpc.py 的 docstring 里。

## 用途
对乌镇集群（112.11.77.146:65091）做 同步/配置/提交/监控/中断/收割 的统一入口。
**agent 应优先用本工具而不是裸写 ssh 命令**——SSH 细节、噪音过滤、sacct 陷阱都在
`hpc_config.json` 里配置好了，且账本机制保证不会误伤集群上其他人的作业。

## 快速上手（在本目录运行）
```bash
python3 hpc.py status            # 活跃 + 近24h终态
python3 hpc.py status 45731722   # 单作业含节点/RSS/Exit
python3 hpc.py submit script.sh --profile probe --array 0-3 --node b10r2n26 --set MAT=mox
python3 hpc.py tail mnz2         # 最近输出的 .out+.err 尾巴
python3 hpc.py results           # 结果行 (C5G7/SUM_MNZ/POOL/ABORT/ladder)
python3 hpc.py cancel 45731722   # 或 cancel all（仅账本内）
python3 hpc.py watch             # 增量监视（cron 用；无变化=零输出）
python3 hpc.py sync up <file|dir> [远端相对名]
python3 hpc.py sync down psn_runs ./local_dir
python3 hpc.py jobs              # 本地账本
python3 hpc.py profile           # 资源档位
```

## 状态文件（勿手工编辑）
- `hpc_config.json` — SSH/档位/notes。改档位前先读 notes。
- `hpc_ledger.json` — submit 自动登记；`cancel all` 与 watch 的作用域。
- `hpc_watch_state.json` — watch 的 diff 基线，原子写。删掉 = 重建基线。

## Agent 决策规则
1. 提交前：`status` 看队列拥堵；内存不确定时先用 `probe` 档位跑探测（如 MNZ/keepall），
   **禁止用大内存正式作业试探资源**（用户明令）。
2. 已知近奇异矩阵问题（strict C5G7 MOX M≥96）：`--node b10r2n26` 钉良节点；
   PARDISO 线程固定 32（`--set PD_THREADS=32`），64 触发 rc=-2。
3. 失败排查顺序：`status <id>` 看 Exit → `tail <前缀>.err` → 若是 rc=-2 先想
   节点 CPU 差异（换良节点重投），不是并发问题（已被串行探针证伪）。
4. `watch` 输出即"值得人看的变化"；返回空 = 一切照旧，直接跳过，不要额外查。
5. 长任务：`submit` 后把 jobid+含义讲清楚再离开；收割逻辑（判 sanity、写 CSV）
   用 Hermes cron agent job，watch 只做状态推送。

## 血泪陷阱（改代码前必读）
- sacct 不认 `-S now-24`，静默返回错误窗口 → 必须显式 `YYYY-MM-DDTHH:MM:SS`，
  且集群时钟是 UTC+8（已封装在 `sacct_lines()`）。
- sacct 加 Elapsed/MaxRSS 列会让 watch 每轮都"有变化" → 状态快照只存 JobID+State。
- 登录 MOTD 有 locale/perl 噪音 → 一切输出过 `NOISE` 正则，自己加 ssh 调用时也要过。
- 本机无 rsync → sync 自动降级 tar-over-ssh；`sync up file 目标名` 支持改名上传。
- SLURM `#SBATCH` 不展开 shell 变量；array 无 task 时日志后缀是 `4294967294`。
- `--mem=0` = 整机独占（节点 1007GB）；profile `probe`=96G/32核 是探针标准档。
- MKL PARDISO 禁并发多实例（锁已在 pardiso_backend.py，脚本别再自己起并发）。

## 与 cron 的关系
- Hermes cron `wuzhen-hpc-watch`（no_agent, 10m）：跑 `hpc_watch.sh`（旧版 bash）。
  要切本工具：script 换 `~/.hermes/scripts/hpc_py_watch.sh`（内容 =
  `cd <本目录> && exec python3 hpc.py watch`）。
- 收割 cron（agent 型）读 `results` 后做 sanity 判定 + 归档 CSV。
