#!/bin/bash
# FIN co-op stage (11-14; part of the approved final plan): arm CB as a 10M co-op fine-tune on top of a
# FIN run's 60M final (60M -> 70M). The train_ppo arguments come from coopstage.py (md5 fc57b219): the FIN stage-2 meta.json
# args in order, then CB's overrides (KL anchor = the 60M start, --bc-kl 0.01 --bc-kl-steps 0, --mix mp2, --band-policy 0, NMA's
# natPlan without coopReplan, CB's carry-nation human-proxy pool, in-run evals solo:ABYSS + coop4h1 96 every 2M, --micro 2048 if the
# base has none). This launcher only adds the host flags (--out --learner-cpus --env-cpus --node --device --mem-frac) and, on a restart,
# --init <newest checkpoint> --start-steps <its steps> (the KL anchor stays the 60M start).
# ext/coop-launch.sh (extension, 11-14) = coop/launch.sh for an EXTENDED checkpoint (coopstage.py on the best extended
# step, start = that step, +10M): takes INIT=<checkpoint> and SRCMETA=<its run's meta.json (EXT)> instead of waiting for SRC/DONE;
# output $RUNS/XC. coopstage.py is FIN's copy (md5 fc57b219). Everything else is identical to coop/launch.sh.
# Usage: SRC=<FIN stage-2 dir> RUNS=<run dir> SEED=<seed offset> GPU=<id> LC=<learner cpus> EC=<env cpus> [MF=<mem frac>] [EMF=<eval mem frac>]
#          bash launch.sh rl      # detached; waits for SRC/DONE, then trains $RUNS/COOP (restart-safe)
#        RUNS=<run dir> bash launch.sh stop
set -u
B=${B:?set B to the work dir}; FIN=${FIN:-$B/FINAL}; T=${T:-$B/code-A6}; MAN=${MAN:-$T/MANIFEST-A6.md5}; MAN_MD5=${MAN_MD5:-2ed55409}
PY=${PY:-$B/prefix/venv/bin/python}; NODE=${NODE:-$B/prefix/node/bin/node}; CS=${CS:-$FIN/coop/coopstage.py}; CS_MD5=${CS_MD5:-cb426b2b}
[ "$(md5sum < $MAN | cut -c1-8)" = "$MAN_MD5" ] || { echo "tree manifest mismatch"; exit 1; }
[ "$(md5sum < $CS | cut -c1-8)" = "$CS_MD5" ] || { echo "coopstage.py md5 mismatch"; exit 1; }
: ${RUNS:?}; O=$RUNS/XC
case ${1:-} in
stop) touch $O/STOP; p=$(cat $O/train.pid 2>/dev/null) && ps -p $p -o args= | grep -q train_ppo && kill $p && echo "killed XC $p"; exit 0;;
rl) [ -n "${_DETACHED:-}" ] || { mkdir -p $O; rm -f $O/STOP; _DETACHED=1 setsid nohup bash $(readlink -f $0) rl > $O/launcher.log 2>&1 < /dev/null & echo "launched, launcher pid $!"; exit 0; };;
*) sed -n 2,12p $0; exit 1;;
esac
: ${SRCMETA:?} ${INIT:?} ${SEED:?} ${GPU:?} ${LC:?} ${EC:?}
echo $$ > $O/launcher.pid
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$B/site PATH=$(dirname $NODE):$PATH SPRL_CUDA_MEM_FRAC=${EMF:-0.01}
st=$($PY -c "import torch; print(int(torch.load('$INIT', map_location='cpu', weights_only=False).get('steps', 0)))")
A0=$O/model-$st.pt
[ -f $O/LINEAGE ] || { cp $INIT $A0; echo "init = KL anchor = $INIT ($st steps, md5 $(md5sum < $A0 | cut -c1-8)) copied as model-$st.pt; coopstage.py $CS_MD5; seed $SEED" > $O/LINEAGE; }
mapfile -t ARGS < <($PY $CS $SRCMETA $A0 $st $SEED)
[ ${#ARGS[@]} -gt 40 ] || { echo "coopstage.py gave ${#ARGS[@]} args"; exit 1; }
[ -f $O/meta.json ] || $PY - "$O/meta.json" "$T" "$MAN" "$A0" "$SRCMETA" "$CS" "${ARGS[@]}" <<'P'
import json, sys, hashlib, socket, time, os
out, tree, man, init, src, cs, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7:]
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
json.dump({'name': 'XC', 'tree': tree, 'manifest': man, 'manifest_md5': md5(man), 'init': init, 'init_md5': md5(init), 'kl_ref': init, 'kl_ref_md5': md5(init),
           'src_meta': src, 'src_meta_md5': md5(src), 'coopstage': cs, 'coopstage_md5': md5(cs),
           'host': socket.gethostname(), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'args': args}, open(out, 'w'), indent=1)
P
newest() { local prev=0 s sz; NI=""; NS=0; for s in $(ls $O | grep -E '^model-[0-9]+\.pt$' | sed -E 's/^model-([0-9]+)\.pt$/\1/' | sort -n); do
  sz=$(stat -c %s $O/model-$s.pt); [ "$sz" -ge $((prev * 95 / 100)) ] && { NI=$O/model-$s.pt; NS=$s; }; prev=$sz; done; }
while true; do
  [ -f $O/STOP ] && { echo "$(date -u '+%F %T') STOP"; exit 0; }
  [ -f $O/DONE ] && exit 0
  newest; echo "$(date -u '+%F %T') XC start from $NI ($NS)"
  (cd $T/tools/rl/py && exec nice -n 19 taskset -c $LC,$EC $PY train_ppo.py "${ARGS[@]}" --init $NI --start-steps $NS \
    --out $O --learner-cpus $LC --env-cpus $EC --node $NODE --device cuda --mem-frac ${MF:-0.06} >> $O/train.log 2>&1) &
  echo $! > $O/train.pid; wait $!; echo "$(date -u '+%F %T') XC train_ppo rc $?"
  grep -q '"event": "end"' $O/train.jsonl 2>/dev/null && [ -f $O/model-final.pt ] && { date -u '+%F %T' > $O/DONE; echo "$(date -u '+%F %T') done"; exit 0; }
  sleep 30
done
