#!/bin/bash
# FINAL (2026-11-13): the approved final training. Two runs, one seed each (a second host runs the second seeds):
#   FIN-MLP-s1 = MLP, the NMA NMA recipe, stage 2 to 60M   (mlp/launch.sh; seed offsets 61000000 / 62000000)
#   FIN-V5-s1  = v5,  the v5 V5-NMA recipe, stage 2 to 60M (v5/launch.sh = the v5 launch.sh + v5/launch.diff; seed offsets 65000000 / 66000000)
# Pipeline from scratch: data (data-plan5, M nat-bot teacher) -> BC (PB5 for the MLP, V5NMA BC for v5) -> stage 1 (5M) -> stage 2 (60M)
#   -> co-op stage (arm's CB: 60M -> 70M, mp2, KL anchor = the 60M start; coop/launch.sh + coop/coopstage.py).
# Usage (B = the work dir; see README.md):
#   bash launch.sh rl          # both runs from the frozen BCs in $ROOT/{mlp,v5}/bc (restart-safe; resumes from the newest checkpoints),
#                                 # plus their co-op stages (coop/launch.sh: each waits for its run's 60M DONE, then 60M -> 70M)
#   bash launch.sh coop        # only the co-op stages (seed offsets 71000000 MLP / 72000000 v5)
#   bash launch.sh stop        # STOP files + kill both trainers by PID
#   bash launch.sh status      # newest checkpoint / last update line per stage
# From scratch into a new root (ROOT=<new dir>, everything is written there; the scripts and frozen trees stay where they are):
#   ROOT=<dir> bash launch.sh data   # re-record data-plan5 with code-N5 (112 single-core workers on CPUS, default 0-55,112-167)
#   ROOT=<dir> bash launch.sh bc     # MLP BC (PB5's command) then v5 BC (the v5 deterministic command), on GPU ${BCGPU:-1}
#   ROOT=<dir> bash launch.sh rl
set -u
B=${B:?set B to the work dir}; FIN=$B/FINAL; ROOT=${ROOT:-$FIN}; PY=$B/prefix/venv/bin/python; NODE=$B/prefix/node/bin/node
D5=$ROOT/data-plan5/act2autochess-75d38e72a6cf; [ -d $D5 ] || D5=$B/N/data-plan5/act2autochess-75d38e72a6cf
MLP=(env F=$ROOT/mlp RUNS=$ROOT/runs/FIN-MLP-s1 SEED1=61000000 SEED2=62000000 GPU=5 LC=0,112 EC=1-27,113-139 PROCS=50 bash $FIN/mlp/launch.sh)
V5=(env F=$ROOT/v5 RUNS=$ROOT/runs/FIN-V5-s1 SEED1=65000000 SEED2=66000000 GPU=6 LC=28,140 EC=29-55,141-167 PROCS=50 bash $FIN/v5/launch.sh)
CM=(env SRC=$ROOT/runs/FIN-MLP-s1/NMA RUNS=$ROOT/runs/FIN-MLP-s1 SEED=71000000 GPU=5 LC=0,112 EC=1-27,113-139 MF=0.06 EMF=0.06 bash $FIN/coop/launch.sh)
CV=(env SRC=$ROOT/runs/FIN-V5-s1/V5NMA RUNS=$ROOT/runs/FIN-V5-s1 SEED=72000000 GPU=6 LC=28,140 EC=29-55,141-167 MF=0.08 EMF=0.01 bash $FIN/coop/launch.sh)
case ${1:-} in
data)
  # = the NMA phase5c.sh step (2) (rec5), on the frozen tree it ran on (code-N5, MANIFEST.md5 ec17d896). Deterministic per seed.
  O=$ROOT/data-plan5; mkdir -p $O/logs; T=$B/code-N5
  [ "$(md5sum < $T/MANIFEST.md5 | cut -c1-8)" = ec17d896 ] || { echo "code-N5 manifest mismatch"; exit 1; }
  export OMP_NUM_THREADS=1 PYTHONPATH=$B/site PATH=$B/prefix/node/bin:$PATH
  CPUS=($(eval echo $(echo ${CPUS:-0-55,112-167} | sed -E 's/([0-9]+)-([0-9]+)/{\1..\2}/g; s/,/ /g')))
  NP='{"teacher":"nat","assign":"mix","mixP":0.5,"live":true,"band":true,"pairS":true,"known":true,"slots":true,"hindsight":true}'
  plan() { local n=$1 cfg=$2 m=$3; for ((k=0;k<n;k++)); do echo "$cfg $m"; done; }
  { plan 72 solo:ABYSS 167; plan 20 solo:HARD 175; plan 6 solo:NORMAL 200; plan 5 coop4l:ABYSS 140; plan 6 coop2l:ABYSS 134; plan 3 coop4ai:ABYSS 50; } > $O/plan.txt
  cd $T/tools/rl; i=0
  while read cfg m; do
    nice -n 19 taskset -c ${CPUS[$((i % ${#CPUS[@]}))]} $NODE record.mjs --out $O --tag $(printf 'w%03d' $i) --matches $m --seed $((90000000 + i * 10000)) --envs 8 \
      --configs $cfg --humans random --rehearsal 0 --boss-rehearsal 0 --obs 3 --nat-plan "$NP" > $O/logs/$(printf 'w%03d' $i).log 2>&1 &
    i=$((i + 1))
  done < $O/plan.txt
  wait; cd $O/act2autochess-75d38e72a6cf && echo "data manifest $(md5sum *.bin *.meta.json *.terms.jsonl | md5sum | cut -c1-8) (reference data-plan5: 0d275a93)";;
bc)
  [ "$ROOT" = "$FIN" ] || { mkdir -p $ROOT/v5; cp -n $FIN/v5/detrun.py $FIN/v5/lookahead.json $ROOT/v5/; }
  env F=$ROOT/mlp DATA=$D5 GPU=${BCGPU:-1} bash $FIN/mlp/launch.sh bc
  env F=$ROOT/v5 DATA=$D5 GPU=${BCGPU:-1} bash $FIN/v5/launch.sh bc;;
rl) "${MLP[@]}" rl; "${V5[@]}" rl; "${CM[@]}" rl; "${CV[@]}" rl;;
coop) "${CM[@]}" rl; "${CV[@]}" rl;;
stop) "${MLP[@]}" stop; "${V5[@]}" stop; "${CM[@]}" stop; "${CV[@]}" stop;;
status)
  for s in FIN-MLP-s1/NM FIN-MLP-s1/NMA FIN-MLP-s1/COOP FIN-V5-s1/V5NM FIN-V5-s1/V5NMA FIN-V5-s1/COOP; do O=$ROOT/runs/$s; [ -d $O ] || continue
    echo "$s: newest $(ls $O | grep -E '^model-[0-9]+\.pt$' | sort -t- -k2 -n | tail -1) $([ -f $O/DONE ] && echo DONE)"
    grep '"event": "update"' $O/train.jsonl 2>/dev/null | tail -1 | cut -c1-300; done;;
*) sed -n 2,16p $0; exit 1;;
esac
