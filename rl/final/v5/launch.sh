#!/bin/bash
# FIN-V5 (2026-11-13): copy of the V5-NMA launch.sh (md5 f6adbe29), patched for the approved final run:
#   seed offsets SEED1/SEED2 (default 65000000 / 66000000), stage 2 to 60M (S2_MAX default 60000000), F default $B/FINAL/v5,
#   DATA overridable (from-scratch re-recording). Diff: launch.diff. Everything else is the v5 file unchanged.
# V5-NMA (arch, 2026-11-13): final candidate = v5 (model5.SPNet5: tokenized entity transformer d128 x 2 layers, 4 heads, unit<->bond
# member bias; separate critic d192 x 1) trained with the NMA recipe at lr 1e-4 (the MLP's NMA ran lr 3e-4):
#   BC on the NMA data-plan5 (M nat-bot teacher, PB5's data) -> stage 1 "V5NM" (NM recipe: solo:HARD hard1, 5M) -> stage 2 "V5NMA"
#   (the ABYSS step: --mix auto1 to 30M). natPlan = NM's {slots, live, assign auto, band, focus, pairS}, band-policy 0, KL ref = the BC.
# Frozen tree code-A6 (MANIFEST-A6.md5 md5 2ed55409) = code-N8 (NMA's code-N5 + N6/N7/N8 flags incl. coopReplan and the co-op
# proxies) + the v5 changes; flags-off parity-gated vs code-N8 and code-A5 ($B/A/gate6/result.txt).
# Usage (restart-safe: re-running a stage resumes from its newest checkpoint; Adam restarts fresh, as in the NMA arm.sh):
#   bash launch.sh bc                                   # host 2 (data-plan5 lives there): deterministic BC -> $F/bc/bc.pt (+ meta.json)
#   GPU=<id> LC=<learner cpus> EC=<env cpus> PROCS=<n> bash launch.sh rl   # stage 1 then stage 2, detached; pid files under $F/runs
#   bash launch.sh stop                                 # touch STOP, kill the trainer by its PID
# Example (NMA's own layout): GPU=7 LC=56,168 EC=57-84,169-194 PROCS=50. LC = 1 core (2 threads); EC = the rest; PROCS ~ #EC threads - 4.
# Test overrides only: S1_MAX, S2_MAX (steps), EVM (in-run eval matches), RUNS (output root).
set -u
B=${B:?set B to the work dir}; F=${F:-$B/FINAL/v5}; T=$B/code-A6; MAN=${MAN:-$T/MANIFEST-A6.md5}; MAN_MD5=${MAN_MD5:-2ed55409}
PY=$B/prefix/venv/bin/python; NODE=$B/prefix/node/bin/node; RUNS=${RUNS:-$F/runs}
DATA=${DATA:-$B/N/data-plan5/act2autochess-75d38e72a6cf}         # the NMA rec5: 112 shards, 1,613,015 samples, manifest md5 0d275a93
CFG='{"d":128,"layers":2,"heads":4,"ffn":2,"member_bias":true,"critic":{"d":192,"layers":1,"heads":4,"ffn":4,"member_bias":false}}'
NM='{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}'
BC=$F/bc/bc.pt
[ "$(md5sum < $MAN | cut -c1-8)" = "$MAN_MD5" ] || { echo "tree manifest mismatch"; exit 1; }
case ${1:-} in
bc)
  mkdir -p $F/bc; [ -s $BC ] && { echo "BC exists: $(md5sum < $BC)"; exit 0; }
  # (deterministic: torch.use_deterministic_algorithms via detrun.py; two replicas on two GPUs gave byte-identical bc.pt)
  $PY - "$F/bc/meta.json" "$T" "$MAN" "$DATA" "$CFG" "$F/detrun.py" "$F/lookahead.json" <<'P'
import json, sys, hashlib, socket, time
out, tree, man, data, cfg, det, member = sys.argv[1:8]
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
json.dump({'tree': tree, 'manifest_md5': md5(man), 'data': data, 'cfg': cfg, 'member_md5': md5(member), 'wrapper_md5': md5(det), 'epochs': 20,
           'host': socket.gethostname(), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, open(out, 'w'), indent=1)
P
  cd $T/tools/rl/py && env CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=$B/site nice -n 19 taskset -c ${CPUS:-101-102,213-214} \
    $PY $F/detrun.py train_bc5.py --data $DATA --out $F/bc --member $F/lookahead.json --cfg "$CFG" --epochs 20 > $F/bc/bc.log 2>&1
  tail -1 $F/bc/bc.log; md5sum $BC; exit 0;;
stop)
  touch $RUNS/STOP; for s in V5NMA V5NM; do p=$(cat $RUNS/$s/train.pid 2>/dev/null) && ps -p $p -o args= | grep -q train_ppo && kill $p && echo "killed $s $p"; done; exit 0;;
rl) [ -n "${_DETACHED:-}" ] || { mkdir -p $RUNS; rm -f $RUNS/STOP; _DETACHED=1 setsid nohup bash $(readlink -f $0) rl > $RUNS/launcher.log 2>&1 < /dev/null & echo "launched, launcher pid $!"; exit 0; };;
*) sed -n 2,16p $0; exit 1;;
esac
# ---- rl (detached) ----
: ${GPU:?} ${LC:?} ${EC:?} ${PROCS:?}
echo $$ > $RUNS/launcher.pid
[ -s $BC ] || { echo "no BC at $BC"; exit 1; }
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$B/site SPRL_CUDA_MEM_FRAC=${EMF:-0.01}
SEED1=${SEED1:-65000000}; SEED2=${SEED2:-66000000}
BASE=(--seed-offset $SEED1 --mix hard1 --curriculum fixed --rehearsal 0 --boss-rehearsal 0
  --reward-v 2 --r2 '{"leak":0.15,"save":0.075,"leakCap":true,"layer":0.002}' --shape-uncapped --shape-tier 0.06 --shape-stack 0.3 --band-policy 0
  --bc-kl 0.01 --bc-kl-steps 0 --ent 0.003 --lr 1e-4 --gamma 0.997 --lam 0.95 --clip 0.2 --target-kl 0.02 --vf 0.5 --vf-warmup 5 --lr-warmup 20 --vclip 0
  --adv-norm global --overlap --legacy-mates 0 --envs 64 --rollout 65536 --minibatch 8192 --epochs 3 --hours 24 --ckpt-every 1e6
  --eval-every 1e6 --eval-at 0 --eval-matches ${EVM:-96} --eval-chunks 4 --eval-configs solo:HARD --log-diag --check
  --amp --micro 1024 --fwd-chunk 2048 --clip-sep --nat-plan "$NM")                         # (v5 memory/stability flags, as V5P6L)
S1=(--max-steps ${S1_MAX:-5000000})
S2=(--seed-offset $SEED2 --mix auto1 --eval-configs solo:HARD,solo:ABYSS --eval-every 2e6 --max-steps ${S2_MAX:-60000000})
newest() { local O=$1 prev=0 st sz; NI=""; NS=0; for st in $(ls $O | grep -E '^model-[0-9]+\.pt$' | sed -E 's/^model-([0-9]+)\.pt$/\1/' | sort -n); do
  sz=$(stat -c %s $O/model-$st.pt); [ "$sz" -ge $((prev * 95 / 100)) ] && { NI=$O/model-$st.pt; NS=$st; }; prev=$sz; done; }
stage() { # stage <name> <init if no checkpoint> <extra args...>
  local name=$1 init0=$2; shift 2; local O=$RUNS/$name; mkdir -p $O
  [ -f $O/DONE ] && return 0
  [ -f $O/meta.json ] || $PY - "$O/meta.json" "$T" "$MAN" "$init0" "$BC" "$name" --kl-ref $BC "${BASE[@]}" --procs $PROCS "$@" <<'P'
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
    (cd $T/tools/rl/py && exec nice -n 19 taskset -c $LC,$EC $PY train_ppo.py --init $init --start-steps $s0 --kl-ref $BC "${BASE[@]}" "${EA[@]}" --procs $PROCS "$@" \
      --out $O --learner-cpus $LC --env-cpus $EC --node $NODE --device cuda --mem-frac ${MF:-0.05} >> $O/train.log 2>&1) &
    echo $! > $O/train.pid; wait $!; echo "$(date -u '+%F %T') $name train_ppo rc $?"
    grep -q '"event": "end"' $O/train.jsonl 2>/dev/null && [ -f $O/model-final.pt ] && { date -u '+%F %T' > $O/DONE; return 0; }
    sleep 30
  done
}
stage V5NM $BC "${S1[@]}"
O2=$RUNS/V5NMA; mkdir -p $O2
st=$($PY -c "import torch; print(int(torch.load('$RUNS/V5NM/model-final.pt', map_location='cpu', weights_only=False).get('steps', 0)))")
[ -f $O2/LINEAGE ] || { cp $RUNS/V5NM/model-final.pt $O2/model-$st.pt; echo "init = V5NM model-final ($st steps) copied as model-$st.pt; KL ref $BC; mix auto1" > $O2/LINEAGE; }
stage V5NMA $O2/model-$st.pt "${S2[@]}"
echo "$(date -u '+%F %T') all stages done"
