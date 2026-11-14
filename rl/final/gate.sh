#!/bin/bash
# FIN gate.sh (11-13): the eval trees reproduce the NMA 30M reference files, per match (seed, win, faShare, nat):
#   g1 code-A6, no pickers, solo:ABYSS 480/28        vs the NMA ev11 NMA-30000000-solo480.json (code-N8)
#   g3 code-A7, c9 pool + xboss-v3, solo:ABYSS        vs the picker run's nma-P9pool-solo.json (code-P10, eval seed set 7000000)
#   g4 code-A7, c9 pool + xboss-v3, coop4h1:ABYSS     vs the picker run's nma-P9pool-coop4h1.json
# g1 + g3 on <cpus A>, g4 on <cpus B>, GPU 7. Result: evals/gate/result.txt
B=${B:?set B to the work dir}; FIN=$B/FINAL; G=$FIN/evals/gate; mkdir -p $G; echo $$ > $G/pid; CA=$1; CB=$2
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 SPRL_CUDA_MEM_FRAC=0.01 PYTHONPATH=$B/site PATH=$B/prefix/node/bin:$PATH PYTHONDONTWRITEBYTECODE=1
PY=$B/prefix/venv/bin/python; NODE=$B/prefix/node/bin/node
NM='{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}'
XP=$(cat $B/X/release/xboss-v3/xplace.json); R=$B/inbox/P-spatial/npool-c9; CK=$B/N/ev11/NMA-30000000.pt
SH=(--bond-log --rehearsal 0 --boss-rehearsal 0 --envs 6 --node $NODE --device cuda --greedy --matches 480 --chunks 28 --band-policy 0 --nat-plan "$NM" --ckpt $CK)
PICK=(--xplace "$XP" --npicker-head $R/c9-js.json --npicker-mode reg --npicker-margin 0.5 --npicker-pool)
COOP=(--mate-rehearsal 3 --human-proxy '{"kinds":["legacy"],"log":true}')
(cd $B/code-A6/tools/rl/py && nice -n 19 taskset -c $CA $PY eval.py --out $G/g1.json "${SH[@]}" --configs solo:ABYSS > $G/g1.log 2>&1
 cd $B/code-A7/tools/rl/py && nice -n 19 taskset -c $CA $PY eval.py --out $G/g3.json "${SH[@]}" --configs solo:ABYSS "${PICK[@]}" > $G/g3.log 2>&1) &
(cd $B/code-A7/tools/rl/py && nice -n 19 taskset -c $CB $PY eval.py --out $G/g4.json "${SH[@]}" --configs coop4h1:ABYSS "${COOP[@]}" "${PICK[@]}" > $G/g4.log 2>&1) &
wait
$PY - > $G/result.txt <<P
import json
def pm(p): return [(m['seed'], m['win'], m['faShare'], m.get('nat')) for c in json.load(open(p))['configs'].values() for m in c['perMatch']]
for g, ref in [('g1', '$B/N/ev11/NMA-30000000-solo480.json'), ('g3', '$FIN/evals/ref/nma-P9pool-solo.json'), ('g4', '$FIN/evals/ref/nma-P9pool-coop4h1.json')]:
    try:
        a, b = pm('$G/' + g + '.json'), pm(ref); d = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        print(g, 'IDENTICAL' if d == 0 else 'DIFFER', 'n', len(a), 'differing', d)
    except Exception as e: print(g, 'ERROR', e)
P
cat $G/result.txt
