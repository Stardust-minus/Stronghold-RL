#!/bin/bash
# FIN-MLP (2026-11-13): the NMA recipe for the MLP (model.py), as actually run by N ($B/N/arm.sh + abyss.sh +
# phase5c.sh, NM.args = --lr 3e-4; runs/NM and runs/NMA meta.json), with stage 2 raised from 30M to 60M:
#   BC "PB5" on the NMA data-plan5 (M nat-bot teacher) -> stage 1 "NM" (solo:HARD --mix hard1, 5M) -> stage 2 "NMA" (--mix auto1 to 60M).
#   natPlan NM = {slots, live, assign auto, band, focus, pairS}, band-policy 0, KL ref = the BC, lr 3e-4.
# The train_ppo argument lists are the NMA, in the NMA order (arm.sh BASE, --procs, the stage extras, --max-steps). Differences to the NMA NM / NMA:
#   tree code-A6 (2ed55409; NM/NMA ran code-N5 ec17d896; A6 = N8 + the flag-gated v5 code, gate6 byte-identical on NM MLP),
#   seed offsets SEED1/SEED2 (default 61000000 / 62000000; the 30M NMA run used 45000000 / 49000000), stage 2 --max-steps 60M (NMA: 30M),
#   --hours 24 (NMA: 12; a wall-clock cap only, so 55M of stage 2 cannot be cut short), outputs under $RUNS.
# Usage (restart-safe: re-running resumes each stage from its newest checkpoint; Adam restarts fresh, as in the NMA arm.sh):
#   bash launch.sh bc                                   # BC as PB5 was trained (not bit-deterministic; the frozen PB5 is $F/bc/bc.pt)
#   GPU=<id> LC=<learner cpus> EC=<env cpus> PROCS=<n> RUNS=<dir> bash launch.sh rl   # stage 1 then stage 2, detached
#   RUNS=<dir> bash launch.sh stop                      # touch STOP, kill the trainer by its PID
# Layout of the NMA NM: GPU=5 LC=0,112 EC=1-27,113-139 PROCS=50.
# Test overrides only: S1_MAX, S2_MAX (steps), EVM (in-run eval matches).
set -u
B=${B:?set B to the work dir}; F=${F:-$B/FINAL/mlp}; T=$B/code-A6; MAN=${MAN:-$T/MANIFEST-A6.md5}; MAN_MD5=${MAN_MD5:-2ed55409}
PY=$B/prefix/venv/bin/python; NODE=$B/prefix/node/bin/node; RUNS=${RUNS:-$F/runs}
DATA=${DATA:-$B/N/data-plan5/act2autochess-75d38e72a6cf}         # the NMA rec5: 112 shards, 1,613,015 samples, manifest md5 0d275a93
NM='{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}'
BC=$F/bc/bc.pt                                                      # PB5 (md5 eaef4a20)
[ "$(md5sum < $MAN | cut -c1-8)" = "$MAN_MD5" ] || { echo "tree manifest mismatch"; exit 1; }
case ${1:-} in
bc)
  mkdir -p $F/bc; [ -s $BC ] && { echo "BC exists: $(md5sum < $BC)"; exit 0; }
  # PB5's command (phase5c.sh step 3; train_bc.py md5 012fab4f is the same file in code-N5 and code-A6). CUDA kernels are not
  # deterministic here, so a re-run gives a statistically equivalent BC, not PB5's bytes.
  cd $T/tools/rl/py && env OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=$B/site nice -n 19 taskset -c ${CPUS:-101-104,213-216} \
    $PY train_bc.py --data $DATA --out $F/bc --epochs 20 --seed 1 > $F/bc/train.log 2>&1
  tail -1 $F/bc/train.log; md5sum $BC; exit 0;;
stop)
  touch $RUNS/STOP; for s in NMA NM; do p=$(cat $RUNS/$s/train.pid 2>/dev/null) && ps -p $p -o args= | grep -q train_ppo && kill $p && echo "killed $s $p"; done; exit 0;;
rl) [ -n "${_DETACHED:-}" ] || { mkdir -p $RUNS; rm -f $RUNS/STOP; _DETACHED=1 setsid nohup bash $(readlink -f $0) rl > $RUNS/launcher.log 2>&1 < /dev/null & echo "launched, launcher pid $!"; exit 0; };;
*) sed -n 2,17p $0; exit 1;;
esac
# ---- rl (detached) ----
: ${GPU:?} ${LC:?} ${EC:?} ${PROCS:?}
echo $$ > $RUNS/launcher.pid
[ -s $BC ] || { echo "no BC at $BC"; exit 1; }
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$B/site PATH=$B/prefix/node/bin:$PATH SPRL_CUDA_MEM_FRAC=0.06
SEED1=${SEED1:-61000000}; SEED2=${SEED2:-62000000}
base() { # the NMA arm.sh BASE with the stage's seed offset
  BASE=(--kl-ref $BC --seed-offset $1 --mix hard1 --curriculum fixed --rehearsal 0 --boss-rehearsal 0
  --reward-v 2 --r2 '{"leak":0.15,"save":0.075,"leakCap":true,"layer":0.002}' --shape-uncapped --shape-tier 0.06 --shape-stack 0.3 --band-policy 0
  --bc-kl 0.01 --bc-kl-steps 0 --ent 0.003 --lr 1e-4 --gamma 0.997 --lam 0.95 --clip 0.2 --target-kl 0.02 --vf 0.5 --vf-warmup 5 --lr-warmup 20 --vclip 0
  --adv-norm global --overlap --legacy-mates 0 --envs 64 --rollout 65536 --minibatch 8192 --epochs 3 --hours 24 --ckpt-every 1e6
  --eval-every 1e6 --eval-at 0 --eval-matches ${EVM:-96} --eval-chunks 4 --eval-configs solo:HARD --log-diag --check); }
S1=(--lr 3e-4 --nat-plan "$NM")                                                                     # NM: NM.args + phase5c.sh
S2=(--mix auto1 --eval-configs solo:HARD,solo:ABYSS --eval-every 2e6 --nat-plan "$NM" --lr 3e-4)     # NMA: abyss.sh + NM.args
newest() { local O=$1 prev=0 st sz; NI=""; NS=0; for st in $(ls $O | grep -E '^model-[0-9]+\.pt$' | sed -E 's/^model-([0-9]+)\.pt$/\1/' | sort -n); do
  sz=$(stat -c %s $O/model-$st.pt); [ "$sz" -ge $((prev * 95 / 100)) ] && { NI=$O/model-$st.pt; NS=$st; }; prev=$sz; done; }
stage() { # stage <name> <init if no checkpoint> <seed offset> <max steps> <extra args...>
  local name=$1 init0=$2 seed=$3 max=$4; shift 4; local O=$RUNS/$name; mkdir -p $O; base $seed
  [ -f $O/DONE ] && return 0
  [ -f $O/meta.json ] || $PY - "$O/meta.json" "$T" "$MAN" "$init0" "$BC" "$name" "${BASE[@]}" --procs $PROCS "$@" --max-steps $max <<'P'
import json, sys, hashlib, socket, time, os
out, tree, man, init, bc, name, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7:]
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
json.dump({'name': name, 'tree': tree, 'manifest': man, 'manifest_md5': md5(man), 'init': init, 'init_md5': md5(init), 'kl_ref': bc, 'kl_ref_md5': md5(bc),
           'host': socket.gethostname(), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'args': args}, open(out, 'w'), indent=1)
P
  while true; do
    [ -f $RUNS/STOP ] && { echo "$(date -u '+%F %T') STOP"; exit 0; }
    newest $O; local init=${NI:-$init0} s0=${NS:-0}
    echo "$(date -u '+%F %T') $name start from $init ($s0)"
    local EA=(); [ "$s0" -gt 0 ] && [ -n "$NI" ] && EA=(--eval-at '')
    (cd $T/tools/rl/py && exec nice -n 19 taskset -c $LC,$EC $PY train_ppo.py --init $init --start-steps $s0 "${BASE[@]}" "${EA[@]}" --procs $PROCS "$@" \
      --max-steps $max --out $O --learner-cpus $LC --env-cpus $EC --node $NODE --device cuda --mem-frac 0.06 >> $O/train.log 2>&1) &
    echo $! > $O/train.pid; wait $!; echo "$(date -u '+%F %T') $name train_ppo rc $?"
    grep -q '"event": "end"' $O/train.jsonl 2>/dev/null && [ -f $O/model-final.pt ] && { date -u '+%F %T' > $O/DONE; return 0; }
    sleep 30
  done
}
stage NM $BC $SEED1 ${S1_MAX:-5000000} "${S1[@]}"
O2=$RUNS/NMA; mkdir -p $O2
st=$($PY -c "import torch; print(int(torch.load('$RUNS/NM/model-final.pt', map_location='cpu', weights_only=False).get('steps', 0)))")
[ -f $O2/LINEAGE ] || { cp $RUNS/NM/model-final.pt $O2/model-$st.pt; echo "init = NM model-final ($st steps) copied as model-$st.pt; KL ref $BC; mix auto1" > $O2/LINEAGE; }
stage NMA $O2/model-$st.pt $SEED2 ${S2_MAX:-60000000} "${S2[@]}"
echo "$(date -u '+%F %T') all stages done"
