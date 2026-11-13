#!/usr/bin/env bash
# The two headline evaluations of docs/RESULTS.md, greedy policy, the policy picks its own band, CPU only.
#   scripts/eval.sh solo480   <ckpt> <out dir>   solo:ABYSS, 480 paired seeds, RL seat r0, at boss rehearsal 3 and 0
#   scripts/eval.sh deploy288 <ckpt> <out dir>   solo / coop4l / coop2l ABYSS, 288 seeds, RL seat r0, rule-AI mates r3, b3
# env: GAME [./Stronghold-Protocol], PYTHON [python3], NODE [node], CHUNKS (parallel node workers) [24], ENVS per worker [6],
#      TAG (output name prefix) [<ckpt dir name>-<ckpt name>],
#      VISIBLE_ONLY: 1 = the deploy information set for the RL seat (env visibleOnly, outputs end in -vis) [1];
#      0 reproduces the tables measured before it existed (docs/RESULTS.md §1-§3)
# Every run of a mode uses the same seeds (solo:ABYSS: 9,000,000 + i), so two checkpoints compare per seed (py/compare.py).
set -euo pipefail
[ $# -eq 3 ] || { sed -n '2,9p' "$0"; exit 2; }
MODE=$1 CK=$2 OUT=$3
GAME="${GAME:-Stronghold-Protocol}" PY="${PYTHON:-python3}" NODE_BIN="${NODE:-node}"
CK="$(cd "$(dirname "$CK")" && pwd)/$(basename "$CK")"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
tag="${TAG:-$(basename "$(dirname "$CK")")-$(basename "$CK" .pt)}"   # e.g. SK-model-30004992
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
cd "$GAME/tools/rl/py"
common=(--ckpt "$CK" --band-policy 1 --greedy --bond-log --rehearsal 0 --chunks "${CHUNKS:-24}" --envs "${ENVS:-6}" --device cpu --node "$NODE_BIN")
sfx=""; if [ "${VISIBLE_ONLY:-1}" = 1 ]; then common+=(--visible-only); sfx=-vis; fi
case "$MODE" in
  solo480)
    for b in 3 0; do
      "$PY" eval.py "${common[@]}" --configs solo:ABYSS --matches 480 --boss-rehearsal "$b" --out "$OUT/$tag-solo480-b$b$sfx.json"
    done ;;
  deploy288)
    "$PY" eval.py "${common[@]}" --configs solo:ABYSS,coop4l:ABYSS,coop2l:ABYSS --matches 288 --boss-rehearsal 3 --mate-rehearsal 3 \
      --out "$OUT/$tag-deploy288$sfx.json" ;;
  *) echo "eval.sh: mode solo480 or deploy288" >&2; exit 2 ;;
esac
