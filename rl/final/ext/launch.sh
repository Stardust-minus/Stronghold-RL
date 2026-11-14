#!/bin/bash
# Extension launcher (11-14; approved "if it is still rising, continue training"; semantics approved
# 11-14 ~02:40 UTC): continue a FIN stage-2 run from its 60M checkpoint to MAX (default 90M) with the SAME stage-2 arguments.
# train_ppo argument list = --init <newest checkpoint> --start-steps <its steps> <stage-2 meta.json "args", in order, with the
# --kl-ref path mapped to this host's md5-checked copy> --eval-at '' --seed-offset <SEED> --auto-smin 0.85 --max-steps <MAX>
# + host flags (--out --learner-cpus --env-cpus --node --device cuda --mem-frac). Last occurrence wins in argparse.
# Continuation semantics (train_ppo checkpoints hold weights only):
#   - Adam restarts fresh; the recipe's own resume warm-up applies (--vf-warmup 5 value-only updates, then --lr-warmup 20 lr ramp).
#   - --auto-smin 0.85: the auto1 mix is pinned at its 60M state (ABYSS .85 = --auto-smax, reached long before 60M with ABYSS
#     rolling win > --auto-target .35). A plain resume would reset ABYSS to 0 and ramp back by .02 per update (~40 updates).
#     The HARD gate still needs ~5 updates of HARD games (N .30 / H .70), which fall inside the value-only warm-up.
#   - --seed-offset SEED (= stage-2 offset + 500000): match seed = base + matches started per process; stage 2 used < 20k per
#     process, so the same offset would replay its seeds from 0. +500k continues on seeds no FIN run has used.
#   - The rolling-episode window (3000) and the minibatch RNG (seed 7) restart.
# Usage: SRCMETA=<stage-2 meta.json> INIT=<60M checkpoint> KLREF=<local copy of the KL-ref BC> RUNS=<run dir> SEED=<offset> ARCH=<mlp|v5>
#        GPU=<id> LC=<learner cpus> EC=<env cpus> [MAX=90000000] [MF=<mem frac>] [EMF=<SPRL_CUDA_MEM_FRAC>] bash launch.sh rl
#        RUNS=<run dir> bash launch.sh stop          (STOP file, then kills the trainer by PID)
# Output $RUNS/EXT: LINEAGE, meta.json (args, md5s, continuation notes), train.{log,jsonl}, model-*.pt every 1M, DONE.
set -u
B=${B:?set B to the work dir}; T=${T:-$B/code-A6}; MAN=${MAN:-$T/MANIFEST-A6.md5}; MAN_MD5=${MAN_MD5:-2ed55409}
PY=${PY:-$B/prefix/venv/bin/python}; NODE=${NODE:-$B/prefix/node/bin/node}
[ "$(md5sum < $MAN | cut -c1-8)" = "$MAN_MD5" ] || { echo "tree manifest mismatch"; exit 1; }
: ${RUNS:?}; O=$RUNS/EXT
case ${1:-} in
stop) touch $O/STOP; p=$(cat $O/train.pid 2>/dev/null) && ps -p $p -o args= | grep -q train_ppo && kill $p && echo "killed EXT $p"; exit 0;;
rl) [ -n "${_DETACHED:-}" ] || { mkdir -p $O; rm -f $O/STOP; _DETACHED=1 setsid nohup bash $(readlink -f $0) rl > $O/launcher.log 2>&1 < /dev/null & echo "launched, launcher pid $!"; exit 0; };;
*) sed -n 2,20p $0; exit 1;;
esac
: ${SRCMETA:?} ${INIT:?} ${KLREF:?} ${SEED:?} ${ARCH:?} ${GPU:?} ${LC:?} ${EC:?}; MAX=${MAX:-90000000}
echo $$ > $O/launcher.pid
if [ $ARCH = v5 ]; then MF=${MF:-0.05}; EMF=${EMF:-0.01}; else MF=${MF:-0.06}; EMF=${EMF:-0.06}; fi
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$B/site PATH=$(dirname $NODE):$PATH SPRL_CUDA_MEM_FRAC=$EMF
KLMD5=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['kl_ref_md5'])" $SRCMETA)
[ "$(md5sum < $KLREF | cut -c1-32)" = "$KLMD5" ] || { echo "KL-ref md5 mismatch ($KLREF vs $KLMD5)"; exit 1; }
st=$($PY -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('steps', 0)))" $INIT)
[ "$st" -gt 0 ] || { echo "no steps in $INIT"; exit 1; }
[ -f $O/LINEAGE ] || { cp $INIT $O/model-$st.pt; echo "init = $INIT ($st steps, md5 $(md5sum < $O/model-$st.pt | cut -c1-8)) copied as model-$st.pt; src meta $SRCMETA (md5 $(md5sum < $SRCMETA | cut -c1-8)); seed $SEED; auto-smin 0.85; max $MAX" > $O/LINEAGE; }
mapfile -t ARGS < <($PY - $SRCMETA $KLREF $SEED $MAX <<'P'
import json, sys
m = json.load(open(sys.argv[1])); a = list(m['args']); kl = sys.argv[2]
for i, x in enumerate(a):
    if x == '--kl-ref': a[i + 1] = kl
    assert x not in ('--init', '--start-steps', '--out', '--auto-smin'), x
a += ['--eval-at', '', '--seed-offset', sys.argv[3], '--auto-smin', '0.85', '--max-steps', sys.argv[4]]
print('\n'.join(a))
P
)
[ ${#ARGS[@]} -gt 80 ] || { echo "only ${#ARGS[@]} args"; exit 1; }
[ -f $O/meta.json ] || $PY - "$O/meta.json" "$T" "$MAN" "$O/model-$st.pt" "$INIT" "$KLREF" "$SRCMETA" "$SEED" "$MAX" "${ARGS[@]}" <<'P'
import json, sys, hashlib, socket, time, os
out, tree, man, init, src_init, kl, srcm, seed, mx, args = *sys.argv[1:10], sys.argv[10:]
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
json.dump({'name': 'EXT', 'tree': tree, 'manifest': man, 'manifest_md5': md5(man), 'init': init, 'init_md5': md5(init), 'init_src': src_init,
           'kl_ref': kl, 'kl_ref_md5': md5(kl), 'src_meta': srcm, 'src_meta_md5': md5(srcm), 'seed_offset': int(seed), 'max_steps': int(mx),
           'continuation': {'optimizer': 'fresh Adam (checkpoints hold weights only); recipe resume warm-up: --vf-warmup 5 value-only updates, then --lr-warmup 20',
                            'mix': '--auto-smin 0.85 pins the auto1 ABYSS share at its 60M state (.85); first ~5 updates N .30 / H .70 until the HARD gate (inside the value warm-up)',
                            'seeds': 'seed offset = stage-2 offset + 500000: unused match seeds (stage 2 used < 20k per process); a same-offset resume would replay them',
                            'other': 'rolling-episode window (3000) and minibatch RNG (seed 7) restart; --eval-at "" (no init eval)'},
           'host': socket.gethostname(), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'args': args}, open(out, 'w'), indent=1)
P
newest() { local prev=0 s sz; NI=""; NS=0; for s in $(ls $O | grep -E '^model-[0-9]+\.pt$' | sed -E 's/^model-([0-9]+)\.pt$/\1/' | sort -n); do
  sz=$(stat -c %s $O/model-$s.pt); [ "$sz" -ge $((prev * 95 / 100)) ] && { NI=$O/model-$s.pt; NS=$s; }; prev=$sz; done; }
while true; do
  [ -f $O/STOP ] && { echo "$(date -u '+%F %T') STOP"; exit 0; }
  [ -f $O/DONE ] && exit 0
  newest; echo "$(date -u '+%F %T') EXT start from $NI ($NS)"
  (cd $T/tools/rl/py && exec nice -n 19 taskset -c $LC,$EC $PY train_ppo.py --init $NI --start-steps $NS "${ARGS[@]}" \
    --out $O --learner-cpus $LC --env-cpus $EC --node $NODE --device cuda --mem-frac $MF >> $O/train.log 2>&1) &
  echo $! > $O/train.pid; wait $!; echo "$(date -u '+%F %T') EXT train_ppo rc $?"
  grep -q '"event": "end"' $O/train.jsonl 2>/dev/null && [ -f $O/model-final.pt ] && { date -u '+%F %T' > $O/DONE; echo "$(date -u '+%F %T') done"; exit 0; }
  sleep 30
done
