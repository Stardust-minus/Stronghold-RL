#!/usr/bin/env bash
# Prepare a Stronghold-Protocol checkout for this repository's RL code:
#   1. clone the game repository (or reuse an existing clone) and check out the pinned commit;
#   2. replace its tools/rl with this repository's rl/;
#   3. apply the server patches in patches/ (gated hooks for the model-AI seats; inert for every other seat).
#
#   ./setup.sh [game dir] [--reset] [--npm]
#     game dir   where the game checkout lives (default: ./Stronghold-Protocol)
#     --reset    discard local changes in an existing checkout first (git reset --hard; removes tools/rl and the patch's new files)
#     --npm      run `npm ci` in the checkout afterwards (the environment itself needs no npm packages; the game's tests do)
#   env: GAME_REPO (clone URL), GAME_COMMIT (default: the commit the patches are made against)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME_REPO="${GAME_REPO:-https://github.com/Stardust-minus/Stronghold-Protocol.git}"
GAME_COMMIT="${GAME_COMMIT:-9d6a1b4e75daf3f77fd00336a5d305d015efec12}"
DEST="" RESET=0 NPM=0
for a in "$@"; do
  case "$a" in
    --reset) RESET=1 ;;
    --npm) NPM=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    -*) echo "setup.sh: unknown option $a" >&2; exit 2 ;;
    *) DEST="$a" ;;
  esac
done
DEST="${DEST:-Stronghold-Protocol}"

if [ ! -d "$DEST/.git" ]; then git clone "$GAME_REPO" "$DEST"; fi
cd "$DEST"
if [ "$RESET" = 1 ]; then
  git reset -q --hard
  rm -rf tools/rl
  for p in "$HERE"/patches/*.patch; do
    sed -n 's|^+++ b/||p' "$p" | while read -r f; do git ls-files --error-unmatch "$f" >/dev/null 2>&1 || rm -f "$f"; done
  done
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "setup.sh: $DEST has local changes to tracked files; commit or stash them, or rerun with --reset" >&2
  exit 1
fi
git cat-file -e "$GAME_COMMIT^{commit}" 2>/dev/null || git fetch -q origin
git checkout -q --detach "$GAME_COMMIT"

rm -rf tools/rl
cp -R "$HERE/rl" tools/rl
for p in "$HERE"/patches/*.patch; do
  git apply --check "$p"
  git apply "$p"
done
if [ "$NPM" = 1 ]; then npm ci; fi
echo "setup.sh: $(pwd) at ${GAME_COMMIT:0:8}, tools/rl from $HERE/rl, patches: $(cd "$HERE/patches" && ls *.patch | tr '\n' ' ')"
