#!/usr/bin/env python3
"""Track C co-op stage (11-14): the train_ppo.py argument list of a 10M co-op fine-tune on top of a finished FIN checkpoint.

  coopstage.py <FIN stage meta.json> <init checkpoint> <start steps> <seed offset> [--check <CB run meta.json>]

Prints one argument per line (FIN's launcher adds the host flags: --out --learner-cpus --env-cpus --node --device --mem-frac).
Recipe = arm CB (from NMA 30M): the FIN arm's own stage arguments (its meta.json, in order: reward, PPO,
lr, model flags such as v5's --amp --micro --fwd-chunk --clip-sep), then these overrides (argparse: the last occurrence wins):
  --init X --kl-ref X        KL anchor = the stage's own start checkpoint (not the BC), --bc-kl 0.01 --bc-kl-steps 0 (persistent)
  --start-steps S --max-steps S+10M   10M steps (no gain at 5M in C1, CA or CB; the gain arrives 5M -> 10M)
  --mix mp2                  solo ~45 % of samples, human + model co-op ~50 %, coop*l ~5 %
  --band-policy 0            legacy picks the bands, as NMA / FIN are trained and evaluated
  --nat-plan NMA's           {slots, live, assign auto, band, focus, pairS}; NO coopReplan (it costs win as an eval rule, E2 / CA)
  --human-proxy CB pool      half the human seats forced onto NMA's co-op carry nations (拉特兰 / 谢拉格; falls back to the route bots' rank
                             order when the nation is off), rest nat:rand / legacy / class / noisy, announce own
  in-run evals solo:ABYSS + coop4h1:ABYSS 96 every 2M (diagnostic only), --micro 2048 if the base has no --micro (memory only).
"""
import json, sys

NMA_NP = '{"slots":true,"live":true,"assign":"auto","band":true,"focus":true,"pairS":true}'
CB_HP = '{"kinds":["mix:nat:lateranoShip=0.25|nat:kjeragShip=0.25|nat:rand=0.2|legacy=0.15|class=0.1|noisy:50=0.05"],"announce":"own"}'
HOST = ('--out', '--learner-cpus', '--env-cpus', '--node', '--device', '--mem-frac', '--init', '--kl-ref', '--start-steps')


def stage_args(base, init, start, seed):
    a = list(base)
    a += ['--init', init, '--kl-ref', init, '--start-steps', str(start), '--max-steps', str(start + 10_000_000),
          '--seed-offset', str(seed), '--mix', 'mp2', '--band-policy', '0', '--nat-plan', NMA_NP, '--human-proxy', CB_HP,
          '--bc-kl', '0.01', '--bc-kl-steps', '0', '--eval-configs', 'solo:ABYSS,coop4h1:ABYSS', '--eval-every', '2e6', '--eval-at', '1e12']
    if '--micro' not in a: a += ['--micro', '2048']
    return a


def last(a):
    r, k = {}, None
    for x in a:
        if x.startswith('--'): k = x; r[k] = True
        else: r[k] = x
    return r


if __name__ == '__main__':
    meta, init, start, seed = sys.argv[1], sys.argv[2], int(float(sys.argv[3])), int(float(sys.argv[4]))
    a = stage_args(json.load(open(meta))['args'], init, start, seed)
    if '--check' in sys.argv:  # effective settings vs a CB-type run (ignores host flags, init, steps, seed, --hours)
        ref = last(json.load(open(sys.argv[sys.argv.index('--check') + 1]))['args']); got = last(a)
        skip = set(HOST) | {'--max-steps', '--seed-offset', '--hours'}
        d = {k: (got.get(k), ref.get(k)) for k in set(got) | set(ref) if k not in skip and got.get(k) != ref.get(k)}
        print(json.dumps(d, ensure_ascii=False, indent=1), file=sys.stderr)
    print('\n'.join(a))
