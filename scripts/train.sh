#!/usr/bin/env bash
# PPO with the current recipe (the m3 / m3s / S line, docs/DESIGN.md): hybrid control, frozen band head, reward version 2
# with the direct layer reward, uncapped potential stacking shaping, KL 0.01 to a fixed anchor, legacy placement r0 / b3.
# Restart-safe: if the run directory already holds model-<steps>.pt checkpoints past <start steps>, the newest complete one
# is resumed instead of <init> (a checkpoint cut short by a kill is skipped).
#
#   scripts/train.sh <run dir> <init ckpt> <kl anchor ckpt> <start steps> [extra train_ppo.py args...]
#
# env (defaults in brackets):
#   GAME     game checkout prepared by setup.sh [./Stronghold-Protocol]
#   PYTHON   [python3]            NODE   [node]          DEVICE  [cuda]
#   MIX      fixed training mix: v2 | v3 | v3s | v3sc | solo1 [v3s]
#   PROCS    environment processes [25]   ENVS  matches per process [64]
#   MAX_STEPS [2e8]   HOURS [24]   EVAL_EVERY [1e7]   SEED_OFFSET (match seed offset of this run) [11000000]
#   LEARNER_CPUS / ENV_CPUS   optional CPU lists (e.g. 0 and 1-24) for the learner and the environment processes
#   BOSS_REHEARSAL legacy boss-round rehearsal in training and the main in-run eval [3]; 0 = the deploy setting
#   VISIBLE_ONLY   1 = the deploy information set for the RL seats (env visibleOnly) in training and in-run evals [0]
# Extra arguments are passed through and win over the defaults above (argparse keeps the last value), e.g. --bc-kl 0.003 (arm SK).
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,19p' "$0"; exit 2; }
OUT=$1 INIT=$2 REF=$3 START=$4; shift 4
GAME="${GAME:-Stronghold-Protocol}"
PY="${PYTHON:-python3}" NODE_BIN="${NODE:-node}"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
INIT="$(cd "$(dirname "$INIT")" && pwd)/$(basename "$INIT")"
REF="$(cd "$(dirname "$REF")" && pwd)/$(basename "$REF")"

# restart-safe resume: the newest model-<steps>.pt that is at least 95 % the size of the one before it
newest="" prev=0
for st in $(ls "$OUT" | sed -nE 's/^model-([0-9]+)\.pt$/\1/p' | sort -n); do
  sz=$(wc -c < "$OUT/model-$st.pt"); [ "$sz" -ge $((prev * 95 / 100)) ] && newest=$st; prev=$sz
done
if [ -n "$newest" ] && [ "$newest" -gt "${START%.*}" ]; then
  echo "$(date '+%F %T') resuming from $OUT/model-$newest.pt (step $newest) instead of $INIT (start $START)"
  INIT="$OUT/model-$newest.pt"; START=$newest
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
cd "$GAME/tools/rl/py"
exec "$PY" train_ppo.py --init "$INIT" --kl-ref "$REF" --start-steps "$START" --seed-offset "${SEED_OFFSET:-11000000}" --out "$OUT" \
  --curriculum fixed --mix "${MIX:-v3s}" --freeze-band --band-policy 0.7 \
  --reward-v 2 --r2 '{"leak":0.15,"save":0.075,"leakCap":true,"layer":0.002}' --reward '{"round":0.05,"fa":1.5}' \
  --shape-tier 0.06 --shape-stack 0.3 --shape-uncapped \
  --rehearsal 0 --boss-rehearsal "${BOSS_REHEARSAL:-3}" --legacy-mates 0.3 --snap-start 0.3 \
  --ent 0.003 --bc-kl 0.01 --bc-kl-steps 0 --lr 1e-4 --rollout 32768 --epochs 3 --minibatch 4096 \
  --vf-warmup 20 --lr-warmup 20 --vclip 0 --adv-norm global --overlap \
  --procs "${PROCS:-25}" --envs "${ENVS:-64}" --max-steps "${MAX_STEPS:-2e8}" --hours "${HOURS:-24}" --ckpt-every 1e6 \
  --eval-every "${EVAL_EVERY:-1e7}" --eval-matches 96 --eval-chunks 4 \
  --eval-configs solo:ABYSS,coop4l:ABYSS,coop2l:ABYSS,coop4ai:ABYSS,solo:HARD,coop4l:HARD,coop4:ABYSS \
  --eval2-configs solo:ABYSS,coop4l:ABYSS,coop2l:ABYSS --eval2-boss-rehearsal 0 \
  ${LEARNER_CPUS:+--learner-cpus "$LEARNER_CPUS"} ${ENV_CPUS:+--env-cpus "$ENV_CPUS"} $([ "${VISIBLE_ONLY:-0}" = 1 ] && echo --visible-only) \
  --check --device "${DEVICE:-cuda}" --node "$NODE_BIN" "$@"
