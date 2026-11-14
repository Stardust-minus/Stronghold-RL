#!/bin/bash
# FIN heldout.sh <cpus> <cand>=<checkpoint> [...] (11-14): held-out confirmation of the top candidates on a seed set never used for
# selection: --seed 200000000 (solo:ABYSS match seeds 202,000,000-202,000,475; coop4h1:ABYSS 780,000,000-780,000,475), setting b
# (c9 normal pool picker + xboss-v3, code-A7), otherwise the headline protocol of evq2.sh (480/28, greedy, r0 b0, --bond-log,
# --band-policy 0 + NM natPlan; co-op --mate-rehearsal 3 + logged legacy human proxy). GPU 7. Outputs evals/heldout/<cand>-b-{solo,coop}.json.
# Restart-safe (finished outputs are skipped). Stop: kill the PID in evals/heldout/pid, then its eval.py.
set -u
B=${B:?set B to the work dir}; FIN=$B/FINAL; H=$FIN/evals/heldout; CP=$1; shift; mkdir -p $H; echo $$ > $H/pid
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 SPRL_CUDA_MEM_FRAC=0.01 PYTHONPATH=$B/site PATH=$B/prefix/node/bin:$PATH PYTHONDONTWRITEBYTECODE=1
PY=$B/prefix/venv/bin/python; NODE=$B/prefix/node/bin/node; T=$B/code-A7
[ "$(md5sum < $T/MANIFEST-A7.md5 | cut -c1-8)" = "${MAN_A7:-cfe0dda0}" ] || { echo "tree manifest mismatch"; exit 1; }
NM='{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}'
XP=$(cat $B/X/release/xboss-v3/xplace.json); R=$B/inbox/P-spatial/npool-c9
SH=(--seed 200000000 --bond-log --rehearsal 0 --boss-rehearsal 0 --envs 6 --node $NODE --device cuda --greedy --matches 480 --chunks 28 --band-policy 0 --nat-plan "$NM"
    --xplace "$XP" --npicker-head $R/c9-js.json --npicker-mode reg --npicker-margin 0.5 --npicker-pool)
for spec in "$@"; do
  cand=${spec%%=*}; ck=${spec#*=}; cp -n $ck $H/$cand.pt; echo "$cand $(md5sum < $H/$cand.pt | cut -c1-8) $ck" >> $H/ckpts.txt
  for cfg in solo coop; do out=$H/$cand-b-$cfg.json; [ -s $out ] && continue
    C=solo:ABYSS; CO=(); [ $cfg = coop ] && { C=coop4h1:ABYSS; CO=(--mate-rehearsal 3 --human-proxy '{"kinds":["legacy"],"log":true}'); }
    echo "$(date -u '+%F %T') $cand $cfg start" >> $H/log
    (cd $T/tools/rl/py && exec nice -n 19 taskset -c $CP $PY eval.py --ckpt $H/$cand.pt --out $out.tmp "${SH[@]}" --configs $C "${CO[@]}" > $H/$cand-b-$cfg.log 2>&1) &
    echo $! > $H/eval.pid; wait $! && mv $out.tmp $out
    echo "$(date -u '+%F %T') $cand $cfg rc $? $(grep -m1 teamWin $H/$cand-b-$cfg.log | sed -E 's/ +/ /g' | cut -c1-80)" >> $H/log
  done
done
echo "$(date -u '+%F %T') done" >> $H/log
