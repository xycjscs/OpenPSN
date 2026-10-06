#!/usr/bin/env bash
# Wuzhen HPC job watcher — prints ONLY state changes since last run.
# Empty stdout = silent (cron delivers nothing). Designed for Hermes cron no_agent.
set -u
KEY=/opt/data/.ssh/hpccube_new.key
SSH="ssh -i $KEY -p 65091 -o StrictHostKeyChecking=no -o ConnectTimeout=20 xycjscs@wuzh02.hpccube.com"
STATE=/opt/data/.hpc_watch_state
GREPNOISE="grep -viE locale|perl|LANG|LC_|setlocale|supported and installed|Falling back"

# jobs to watch (r16 family + mnz probes)
JOBS="45726545 45728306 45729985 45731061 45731062"

snap=$($SSH '
sacct -S 2026-10-02T12:00:00 -u xycjscs --noheader -X --parsable2 --format=JobID,State 2>/dev/null | grep -vE "\.batch|\.extern";
echo "###RESULTS###";
grep -h -E "C5G7|SUM_MNZ|POOL TOTAL|ABORT gate" /work/home/xycjscs/psn_runs/r16_*.out /work/home/xycjscs/psn_runs/mnz_*.out 2>/dev/null;
echo "###FAILS###";
for e in /work/home/xycjscs/psn_runs/r16_*.err /work/home/xycjscs/psn_runs/mnz_*.err; do
  [ -s "$e" ] && grep -l -E "Error|FAIL" "$e" 2>/dev/null | while read f; do
    echo "$f :: $(grep -E "Error|FAIL" "$f" | tail -1 | cut -c1-120)";
  done;
done' 2>/dev/null | grep -viE 'locale|perl|LANG|LC_|setlocale|supported and installed|Falling back')

if [ -z "$snap" ]; then
  echo "WATCHER: ssh to cluster failed $(date +%H:%M)"
  exit 0
fi

if [ ! -f "$STATE" ]; then
  printf '%s' "$snap" > "$STATE"
  echo "HPC监视器已启动：跟踪作业 $(echo $JOBS | tr '\n' ' ')，此后仅在状态变化或出结果时通知。"
  exit 0
fi

new=$(diff <(cat "$STATE") <(printf '%s' "$snap") | grep '^>' | sed 's/^> //' || true)
printf '%s' "$snap" > "$STATE"

[ -z "$new" ] && exit 0

# quiet bookkeeping lines: cgroup/RUNNING churn we already know
echo "$new" | grep -vE '^[0-9]+\.ba|^[0-9]+\.ex' | grep -q . || exit 0

echo "【乌镇作业更新】$(TZ=Asia/Shanghai date '+%m-%d %H:%M') 北京时间"
echo "$new" | grep -vE '^[0-9]+\.ba|^[0-9]+\.ex'
