#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hpc.py — 乌镇(SLURM)集群调用工具：同步 / 配置 / 提交 / 监控 / 中断 / 收割。

单文件、仅标准库（rsync 走 subprocess）。配置在 hpc_config.json（同目录），
SSH 细节只写一次，子命令全部读它。

用法速查
  hpc.py status                    全部在跑/排队作业 + 每任务状态
  hpc.py status <jobid>            单作业（含各 array task、节点、RSS）
  hpc.py sync up   <本地路径> [远端相对路径]     增量上传（rsync，不删远端）
  hpc.py sync down <远端路径> [本地目录]         增量下载
  hpc.py submit <脚本> [--profile P] [--array 0-3] [--node b10r2n26] [--set K=V]
                                   脚本先自动 sync up，再 sbatch，打印 jobid
  hpc.py tail <模式> [-n 20]       grep 远端输出（模式如 r16、mnz_45729985）
  hpc.py results [模式]            抓结果行（C5G7/SUM_MNZ/POOL，可自定义）
  hpc.py cancel <jobid|all>        scancel（all = 本工具账本内全部未终态作业）
  hpc.py watch [--loop N]          增量状态监视（配 cron 用；无变化=无输出）
  hpc.py jobs                      本地账本（提交过的作业）
  hpc.py profile                   列出资源配置档位

账本: hpc_ledger.json — submit 自动登记 jobid→{name,script,profile,array,node}，
cancel all / watch / status 都以账本为范围，不碰集群上别人的作业。
状态快照: hpc_watch_state.json — watch 的 diff 基线（原子写）。

设计约束（血泪，勿改）:
  * sacct 只取 JobID,State —— Elapsed/MaxRSS 每轮变，进 diff 就是假警报。
  * 输出统一过滤 locale/perl 噪音（集群登录 MOTD）。
  * scp 不可用时 rsync over ssh；远端路径一律相对 $HOME。
"""
import argparse, json, os, re, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "hpc_config.json")
LEDGER_PATH = os.path.join(HERE, "hpc_ledger.json")
STATE_PATH = os.path.join(HERE, "hpc_watch_state.json")
NOISE = re.compile(r"locale|perl|LANG|LC_|setlocale|supported and installed|Falling back", re.I)

def load_cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return json.load(f)

def ssh_base(cfg):
    c = cfg["ssh"]
    return ["ssh", "-i", c["key"], "-p", str(c["port"]),
            "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
            f"{c['user']}@{c['host']}"]

def run(cmd, timeout=120):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if not NOISE.search(l))
    return p.returncode, out

def rsh(cfg, remote_cmd, timeout=120):
    return run(ssh_base(cfg) + [remote_cmd], timeout=timeout)

# ---------------------------------------------------------------- ledger --
def ledger_load():
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def ledger_save(d):
    tmp = LEDGER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LEDGER_PATH)

# ----------------------------------------------------------------- status --
def sacct_lines(cfg, hours=24):
    # 集群 sacct 不认 "now-24"（静默退化为 since-now，只剩活跃作业）——
    # 必须传显式时间戳（UTC+8 集群本地时间）。
    since = time.strftime("%Y-%m-%dT%H:%M:%S",
                          time.gmtime(time.time() - hours * 3600 + 8 * 3600))
    rc, out = rsh(cfg, f"sacct -S {since} -u $(whoami) --noheader -X --parsable2 "
                       f"--format=JobID,State,Exit 2>/dev/null | grep -vE '\\.batch|\\.extern'")
    return [l for l in out.splitlines() if l.strip()]

def cmd_status(cfg, args):
    if args.job:
        rc, out = rsh(cfg, f"sacct -j {args.job} --noheader -X --parsable2 "
                           f"--format=JobID,State,Exit,NodeList,MaxRSS,Elapsed 2>/dev/null")
        print(out or "(无记录)")
        rc, out = rsh(cfg, f"squeue -j {args.job} 2>/dev/null")
        if out.strip():
            print("--- 排队中 ---"); print(out)
        return
    rc, out = rsh(cfg, "squeue -u $(whoami) -o '%.10i %.9j %.2t %.10M %.6D %.20R' 2>/dev/null")
    print("== 活跃 ==");  print(out or "(无)")
    print("== 近24h终态 ==")
    done = [l.split("|")[0] + " " + l.split("|")[1] for l in sacct_lines(cfg)
            if l.split("|")[1] not in ("RUNNING", "PENDING", "CONFIGURING")]
    for l in done[-30:]:
        print(l)

# ------------------------------------------------------------------- sync --
def cmd_sync(cfg, args):
    c = cfg["ssh"]
    rbase = c.get("remote_root", ".")
    ssh_cmd = "ssh -i %s -p %d -o StrictHostKeyChecking=no -o ConnectTimeout=20" % (
        c["key"], c["port"])
    try:
        has_rsync = subprocess.run(["which", "rsync"], capture_output=True).returncode == 0 \
            and rsh(cfg, "which rsync")[1].strip()
    except Exception:
        has_rsync = False
    if has_rsync:
        dst_user = f"{c['user']}@{c['host']}"
        if args.dir == "up":
            src = args.path
            rel = args.dest or os.path.basename(src.rstrip("/"))
            dst = f"{dst_user}:{rbase}/{rel}" if not rel.startswith("/") else f"{dst_user}:{rel}"
        else:
            src = f"{dst_user}:{args.path if args.path.startswith('/') else rbase + '/' + args.path}"
            dst = args.dest or "./hpc_downloads"
            os.makedirs(dst, exist_ok=True)
        cmd = ["rsync", "-az", "--info=stats1", "-e", ssh_cmd]
        if args.delete:
            cmd.append("--delete")
        cmd += [src, dst]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(p.stdout.strip() or p.stderr[-300:])
        return p.returncode
    # ---- tar-over-ssh 兜底（任一端无 rsync）----
    tgt = f"{c['user']}@{c['host']}"
    if args.dir == "up":
        src = args.path
        rel = (args.dest or os.path.basename(src.rstrip("/")))
        rdir = os.path.dirname(rel) or "."
        rsh(cfg, f"mkdir -p {rbase}/{rdir}")
        base = os.path.basename(src)
        if os.path.isdir(src):
            tar = ["tar", "-czf", "-", "-C", src, "."]
            rd = f"{rbase}/{rel}"
            rsh(cfg, f"mkdir -p {rd}")
            extract = f"tar -xzf - -C {rd}"
        else:
            tar = ["tar", "-czf", "-", "-C", os.path.dirname(src) or ".", base]
            # src 是本地文件 -> 目标默认就是文件名（有无 dest 都一样）
            if True:
                tmpdir = f"{rbase}/{rdir}/.hpc_sync_tmp"
                rd = tmpdir
                extract = f"mkdir -p {tmpdir} && tar -xzf - -C {tmpdir} && mv -f {tmpdir}/{base} {rbase}/{rel} && rmdir {tmpdir}"
            else:
                rd = f"{rbase}/{rel}"
                extract = f"mkdir -p {rd} && tar -xzf - -C {rd}"
        p1 = subprocess.Popen(tar, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(ssh_cmd.split() + [tgt, extract],
                              stdin=p1.stdout, stderr=subprocess.PIPE, text=True)
        p1.stdout.close(); p2.communicate(timeout=900)
        print(f"uploaded {src} -> {tgt}:{rbase}/{rel}" if "." in os.path.basename(rel)
              else f"uploaded {src} -> {tgt}:{rd}/")
        return p2.returncode
    else:
        rp = args.path if args.path.startswith("/") else f"{rbase}/{args.path}"
        dst = args.dest or "./hpc_downloads"
        os.makedirs(dst, exist_ok=True)
        p2 = subprocess.Popen(ssh_cmd.split() + [tgt, f"tar -czf - -C {rp} ."],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p1 = subprocess.Popen(["tar", "-xzf", "-", "-C", dst], stdin=p2.stdout)
        p2.stdout.close(); p1.communicate(timeout=900)
        print(f"downloaded {tgt}:{rp} -> {dst}")
        return p1.returncode

# ----------------------------------------------------------------- submit --
def cmd_submit(cfg, args):
    prof = cfg["profiles"].get(args.profile or cfg["default_profile"])
    script = args.script
    if not os.path.exists(script):
        print(f"本地脚本不存在: {script}", file=sys.stderr); return 2
    # 1) 同步脚本（及其旁边同名 .py 依赖不管——约定脚本自包含或引用集群仓库）
    c = cfg["ssh"]; rbase = c.get("remote_root", ".")
    rdir = f"{rbase}/{args.remote_dir}" if args.remote_dir else rbase
    rsh(cfg, f"mkdir -p {rdir}")
    rc, out = run(["rsync", "-az", "-e", f"ssh -i {c['key']} -p {c['port']} "
                   f"-o StrictHostKeyChecking=no", script,
                   f"{c['user']}@{c['host']}:{rdir}/"], timeout=300)
    if rc != 0:
        print("rsync 失败:", out, file=sys.stderr); return 1
    # 2) 包一层 sbatch 覆写（profile/指定节点/array 都在命令行给，不改脚本本体）
    rscript = f"{rdir}/{os.path.basename(script)}"
    parts = [f"export HPC_PROFILE={args.profile or cfg['default_profile']}",
             f"sbatch --export=ALL,HPC_PROFILE={args.profile or cfg['default_profile']}"]
    for k in ("partition", "nodes", "cpus", "mem", "time"):
        if k in prof:
            flag = {"partition": "-p", "nodes": "-N", "cpus": "-c",
                    "mem": "--mem", "time": "-t"}[k]
            parts.append(f"{flag} {prof[k]}")
    if args.array:
        parts.append(f"--array={args.array}")
    if args.node:
        parts.append(f"--nodelist={args.node}")
    for kv in args.set or []:
        parts.insert(1, f"export {kv}")
    parts.append(f"{rscript}")
    wrapped = "; ".join(parts)
    rc, out = rsh(cfg, wrapped)
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        print("提交失败:", out, file=sys.stderr); return 1
    job = m.group(1)
    led = ledger_load()
    led[job] = {"name": os.path.basename(script), "script": rscript,
                "profile": args.profile or cfg["default_profile"],
                "array": args.array or "", "node": args.node or "",
                "ts": time.strftime("%Y-%m-%d %H:%M")}
    ledger_save(led)
    print(f"job {job} 已提交  (profile={parts[1]})" if False else f"job {job} 已提交（{args.profile or cfg['default_profile']}）")
    if args.node:
        print(f"钉节点: {args.node}")

# ------------------------------------------------------------------ tail --
def cmd_tail(cfg, args):
    base = cfg["runs_glob"]
    globs = (base.rsplit(".", 1)[0] + ".{out,err}" if ".*" in base or base.endswith(".out")
             else base)
    g = globs.replace("*{out,err}", args.pattern + "*.{out,err}") if "*" in globs else base
    rc, out = rsh(cfg, f"for f in $(ls -t {g} 2>/dev/null | head -6); do echo \"== $f\"; tail -{args.n} $f; done")
    print("\n".join(l for l in out.splitlines() if not NOISE.search(l)))

def cmd_results(cfg, args):
    pat = args.pattern or "C5G7|SUM_MNZ|POOL TOTAL|ABORT gate|ladder"
    rc, out = rsh(cfg, f"grep -h -E '{pat}' {cfg['runs_glob']} 2>/dev/null | sort -u | tail -40")
    print(out or "(暂无结果行)")

# ----------------------------------------------------------------- cancel --
def cmd_cancel(cfg, args):
    led = ledger_load()
    if args.job == "all":
        live = []
        rc, out = rsh(cfg, "squeue -u $(whoami) -h -o %A 2>/dev/null")
        live = set(out.split())
        targets = [j for j in led if j in live]
        if not targets:
            print("账本内无活跃作业"); return
    else:
        targets = [args.job]
    rc, out = rsh(cfg, "scancel " + " ".join(shlex.quote(t) for t in targets))
    print(f"已 scancel: {' '.join(targets)}")

# ------------------------------------------------------------------ watch --
def cmd_watch(cfg, args):
    """增量状态推送。无变化 → 无输出（cron 静默）。"""
    rc, out = rsh(cfg, "squeue -u $(whoami) -h -o '%A|%j|%t' 2>/dev/null; echo '###END###'; "
                        + f"grep -h -E 'C5G7|SUM_MNZ|POOL TOTAL|ABORT gate' {cfg['runs_glob']} 2>/dev/null")
    lines = [l for l in out.splitlines() if l.strip() and not NOISE.search(l)]
    try:
        end = lines.index("###END###"); active = lines[:end]; results = lines[end+1:]
    except ValueError:
        active, results = lines, []
    state = {"active": sorted(active), "results": sorted(set(results))}
    old = {}
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, STATE_PATH)
    if not old:
        print(f"HPC watch 基线建立：{len(active)} 个活跃作业。之后仅变化时通知。")
        return
    gone = [a for a in old["active"] if a not in state["active"]]
    newr = [r for r in state["results"] if r not in set(old["results"])]
    msgs = []
    if gone:
        ids = [g.split("|")[0] for g in gone]
        rc, out2 = rsh(cfg, f"sacct -j {','.join(ids)} -X --noheader --parsable2 "
                            f"--format=JobID,State,MaxRSS 2>/dev/null | grep -vE '\\.batch|\\.extern'")
        msgs.append("作业结束:\n" + (out2 or "\n".join(gone)))
    if newr:
        msgs.append("新结果:\n" + "\n".join(newr))
    if msgs:
        print("【乌镇】" + "\n".join(msgs))

def cmd_jobs(cfg, args):
    led = ledger_load()
    if not led:
        print("(账本为空)"); return
    for j, info in led.items():
        print(f"{j}  {info.get('ts','')}  {info['name']:<18} profile={info['profile']:<12} "
              f"array={info['array'] or '-':<6} node={info['node'] or '-'}")

def cmd_profile(cfg, args):
    for name, p in cfg["profiles"].items():
        star = "*" if name == cfg["default_profile"] else " "
        print(f"{star} {name}: " + "  ".join(f"{k}={v}" for k, v in p.items()))

def main():
    ap = argparse.ArgumentParser(prog="hpc.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.add_argument("job", nargs="?")
    s = sub.add_parser("sync"); s.add_argument("dir", choices=["up", "down"])
    s.add_argument("path"); s.add_argument("dest", nargs="?")
    s.add_argument("--delete", action="store_true")
    s = sub.add_parser("submit"); s.add_argument("script")
    s.add_argument("--profile"); s.add_argument("--array"); s.add_argument("--node")
    s.add_argument("--set", action="append", metavar="K=V")
    s.add_argument("--remote-dir", default="psn_strict")
    s = sub.add_parser("tail"); s.add_argument("pattern"); s.add_argument("-n", type=int, default=20)
    s = sub.add_parser("results"); s.add_argument("pattern", nargs="?")
    s = sub.add_parser("cancel"); s.add_argument("job")
    s = sub.add_parser("watch"); s.add_argument("--loop", type=int, default=0)
    sub.add_parser("jobs"); sub.add_parser("profile")
    args = ap.parse_args()
    cfg = load_cfg()
    fn = {"status": cmd_status, "sync": cmd_sync, "submit": cmd_submit,
          "tail": cmd_tail, "results": cmd_results, "cancel": cmd_cancel,
          "watch": cmd_watch, "jobs": cmd_jobs, "profile": cmd_profile}[args.cmd]
    rc = fn(cfg, args)
    sys.exit(rc or 0)

if __name__ == "__main__":
    main()
