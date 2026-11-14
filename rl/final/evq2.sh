#!/bin/bash
# FIN evq2.sh <lane> <cpus> [gpu] (11-14; replaces evq.sh, same jobs and protocol): milestone evals of the FINAL runs, driven by
# $E/runs.txt (re-read every pass; lines "<run> <stage dir> <milestone M ...>", e.g. "FIN-MLP-s1 $FIN/runs/FIN-MLP-s1/NMA 30 40 50 60").
# Checkpoint of (run, M): $E/pin/<run>-<M>M (a file holding a checkpoint path; used for relayed a second host checkpoints), else the first
# complete checkpoint >= M million steps in the stage dir (complete = a newer one exists or the stage is DONE), else model-final.pt
# when DONE. Jobs: settings a (no pickers, code-A6) and b (c9 normal pool picker + xboss-v3, code-A7) x solo:ABYSS / coop4h1:ABYSS,
# 480/28 seeds, headline protocol (greedy, r0 b0, --bond-log, --band-policy 0 + NM natPlan; co-op --mate-rehearsal 3 + logged legacy
# human proxy). Jobs run in milestone order across runs. Lanes share jobs through claim dirs ($E/claims/<job>/{pid,eval.pid});
# GUARD=1: claim only while the lane's CPUs are idle (< 25 % busy over 3 s). Restart-safe. Stop: touch $E/STOP-<lane> (after the
# current job) or kill the lane PID ($E/lane-<lane>.pid) and its eval.py ($E/claims/<job>/eval.pid).
set -u
B=${B:?set B to the work dir}; FIN=$B/FINAL; E=$FIN/evals; LANE=$1; CP=$2; GPU=${3:-7}
mkdir -p $E/ckpt $E/claims $E/out $E/pin; echo $$ > $E/lane-$LANE.pid; rm -f $E/STOP-$LANE
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU SPRL_CUDA_MEM_FRAC=0.01 PYTHONPATH=$B/site PATH=$B/prefix/node/bin:$PATH PYTHONDONTWRITEBYTECODE=1
PY=$B/prefix/venv/bin/python; NODE=$B/prefix/node/bin/node
NM='{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}'
XP=$(cat $B/X/release/xboss-v3/xplace.json); R=$B/inbox/P-spatial/npool-c9
SH=(--bond-log --rehearsal 0 --boss-rehearsal 0 --envs 6 --node $NODE --device cuda --greedy --matches 480 --chunks 28 --band-policy 0 --nat-plan "$NM")
PICK=(--xplace "$XP" --npicker-head $R/c9-js.json --npicker-mode reg --npicker-margin 0.5 --npicker-pool)
COOP=(--mate-rehearsal 3 --human-proxy '{"kinds":["legacy"],"log":true}')
declare -A TREE=([a]=$B/code-A6 [b]=$B/code-A7)
[ "$(md5sum < $B/code-A6/MANIFEST-A6.md5 | cut -c1-8)" = "${MAN_A6:-2ed55409}" ] && [ "$(md5sum < $B/code-A7/MANIFEST-A7.md5 | cut -c1-8)" = "${MAN_A7:-cfe0dda0}" ] || { echo "tree manifest mismatch"; exit 1; }
log() { echo "$(date -u '+%F %T') [$LANE] $*" >> $E/log; }
steps() { ls $1 2>/dev/null | grep -E '^model-[0-9]+\.pt$' | sed -E 's/^model-([0-9]+)\.pt$/\1/' | sort -n; }
ckpt() { # <run> <stage dir> <M>: CK / ST
  local run=$1 O=$2 M=$3 st newer; CK=""; ST=""
  if [ -s $E/pin/$run-${M}M ]; then CK=$(cat $E/pin/$run-${M}M); [ -s "$CK" ] || { CK=""; return 0; }
    ST=$(basename $CK .pt | sed 's/^model-//'); return 0; fi
  [ -d $O ] || return 0
  st=$(steps $O | awk -v t=$((M * 1000000)) '$1 >= t' | head -1)
  if [ -n "$st" ]; then newer=$(steps $O | awk -v s=$st '$1 > s' | head -1); { [ -n "$newer" ] || [ -f $O/DONE ]; } && { CK=$O/model-$st.pt; ST=$st; }
  elif [ -f $O/DONE ] && [ -f $O/model-final.pt ]; then CK=$O/model-final.pt; ST=final; fi; }
busy() { python3 - "$CP" <<'P'
import sys, time
cpus = set()
for part in sys.argv[1].split(','):
    a, _, b = part.partition('-'); cpus.update(range(int(a), int(b or a) + 1))
def snap():
    r = {}
    for l in open('/proc/stat'):
        f = l.split()
        if f[0].startswith('cpu') and f[0][3:].isdigit() and int(f[0][3:]) in cpus:
            v = list(map(int, f[1:])); r[f[0]] = (sum(v), v[3] + v[4])
    return r
a = snap(); time.sleep(3); b = snap()
print(int(100 * sum(1 - (b[k][1] - a[k][1]) / max(1, b[k][0] - a[k][0]) for k in a) / max(1, len(a))))
P
}
alive() { [ -n "$1" ] && kill -0 $1 2>/dev/null; }
while true; do
  [ -f $E/STOP-$LANE ] && { log "STOP"; exit 0; }
  left=0; skip=0
  # (milestone, run, stage dir) in milestone order
  while read -r M run O; do
    ckpt $run $O $M; [ -z "$CK" ] && { left=1; continue; }
    for set in a b; do for cfg in solo coop; do
      job=$run-${M}M-$set-$cfg; out=$E/out/$job.json
      [ -s $out ] || [ -f $out.failed ] && continue
      left=1
      [ $skip = 1 ] && continue
      if [ "${GUARD:-0}" = 1 ]; then u=$(busy); [ "$u" -lt 25 ] || { log "lane busy ${u}%: waiting"; skip=1; continue; }; fi
      if ! mkdir $E/claims/$job 2>/dev/null; then
        p=$(cat $E/claims/$job/pid 2>/dev/null) || continue
        { alive $p || alive "$(cat $E/claims/$job/eval.pid 2>/dev/null)"; } && continue
        rm -rf $E/claims/$job; mkdir $E/claims/$job 2>/dev/null || continue
      fi
      echo $$ > $E/claims/$job/pid
      CKC=$E/ckpt/$run-${M}M-$ST.pt; [ -s $CKC ] || { cp $CK $CKC.tmp$$ && mv $CKC.tmp$$ $CKC; }
      echo "$ST $(md5sum < $CKC | cut -c1-8) $CK" > $E/out/$job.ckpt
      C=solo:ABYSS; CO=(); [ $cfg = coop ] && { C=coop4h1:ABYSS; CO=("${COOP[@]}"); }
      EX=(); [ $set = b ] && EX=("${PICK[@]}")
      log "$job start: $ST ($CK), tree ${TREE[$set]}, cpus $CP"; t0=$(date +%s)
      (cd ${TREE[$set]}/tools/rl/py && exec nice -n 19 taskset -c $CP $PY eval.py --ckpt $CKC --out $out.tmp "${SH[@]}" --configs $C "${CO[@]}" "${EX[@]}" > $E/out/$job.log 2>&1) &
      echo $! > $E/claims/$job/eval.pid; wait $!; rc=$?
      if [ $rc = 0 ] && [ -s $out.tmp ]; then mv $out.tmp $out; else n=$(cat $E/out/$job.nfail 2>/dev/null || echo 0); echo $((n + 1)) > $E/out/$job.nfail; [ $n -ge 1 ] && touch $out.failed; fi
      log "$job rc $rc $(( $(date +%s) - t0 )) s | $(grep -m1 teamWin $E/out/$job.log | sed -E 's/ +/ /g' | cut -c1-90)"
      rm -rf $E/claims/$job
      $PY $FIN/finsum.py > $E/summary.md 2>> $E/finsum.err
      [ -f $E/STOP-$LANE ] && { log "STOP"; exit 0; }
    done; done
  done < <(grep -v '^#' $E/runs.txt | awk 'NF >= 3 { for (i = 3; i <= NF; i++) print $i, $1, $2 }' | sort -n -k1,1 -s)
  [ $left = 0 ] && [ -f $E/ALLDONE ] && { log "all milestone evals done"; exit 0; }
  [ $skip = 1 ] && sleep 180; sleep 60
done
