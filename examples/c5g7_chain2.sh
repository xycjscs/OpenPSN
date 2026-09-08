#!/bin/bash
# C5G7 续链：等单组件 M16_S6 (PID 48012) 完成 → 1/4 芯 M16_S2 → M16_S3
# 注意: probe 参数顺序是 (S, M)！
set -u
cd /opt/data/workspace/OpenPSN
LOG=examples/c5g7_sweep.log
ASMPID=48012

while kill -0 $ASMPID 2>/dev/null; do sleep 30; done
echo "TS $(date '+%F %T') asm_M16_S6 done, starting qc M16_S2 (S=2 M=16)" >> $LOG
free -m | head -2 >> $LOG

python3 -u examples/c5g7_probe.py 2 16 1e-10 > examples/c5g7_qc_M16_S2.log 2>&1
RC=$?
echo "TS $(date '+%F %T') qc M16_S2 done rc=$RC" >> $LOG
free -m | head -2 >> $LOG

if [ $RC -eq 137 ]; then
  echo "TS $(date '+%F %T') qc M16_S3 SKIPPED (S2 was OOM-killed)" >> $LOG
elif [ $(free -m | awk '/Mem:/{print $7}') -lt 30000 ]; then
  echo "TS $(date '+%F %T') qc M16_S3 SKIPPED (insufficient RAM)" >> $LOG
else
  python3 -u examples/c5g7_probe.py 3 16 1e-10 > examples/c5g7_qc_M16_S3.log 2>&1
  echo "TS $(date '+%F %T') qc M16_S3 done rc=$?, CHAIN COMPLETE" >> $LOG
fi
