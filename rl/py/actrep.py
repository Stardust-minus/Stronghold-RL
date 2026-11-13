"""Action histogram report (env actLog): per RL seat per match. python actrep.py run.json [--vs off.json  (bounty-field A/B, paired)]"""
import json, sys
from collections import defaultdict
import numpy as np

def load(p):
    return json.load(open(p))['configs']

def seats(c):
    for m in c['perMatch']:
        for s in m.get('act') or []:
            if not s.get('none'): yield m, s

def mean(xs): xs = list(xs); return float(np.mean(xs)) if xs else float('nan')

def report(cfg, c):
    S = list(seats(c)); n = len(S)
    print(f'== {cfg}: {len(c["perMatch"])} matches, {n} RL seats, team win {c["teamWin"]:.3f}')
    g = lambda f: mean(f(s) for _, s in S)
    print(f'  per seat-match: refresh {g(lambda s: s["refresh"]):.1f} | level {g(lambda s: s["level"]):.1f} | freeze {g(lambda s: s["freeze"]):.2f}'
          f' | buys chess {g(lambda s: s["buy"]["chess"]):.1f} item {g(lambda s: s["buy"]["item"]):.2f} other {g(lambda s: s["buy"]["other"]):.2f}'
          f' | sells chess {g(lambda s: s["sell"]["chess"]):.1f} item {g(lambda s: s["sell"]["item"]):.2f}'
          f' | reward picks chess {g(lambda s: s["reward"]["chess"]):.2f} item {g(lambda s: s["reward"]["item"]):.2f} other {g(lambda s: s["reward"]["other"]):.2f}')
    print(f'  share of seat-matches: refresh 0 {mean(s["refresh"] == 0 for _, s in S):.2f}, refresh >= 10 {mean(s["refresh"] >= 10 for _, s in S):.2f};'
          f' bought >= 1 item {mean(s["buy"]["item"] >= 1 for _, s in S):.2f}; levelled 0 {mean(s["level"] == 0 for _, s in S):.2f}')
    st = lambda k: mean((s.get('stats') or {}).get(k) or 0 for _, s in S)
    print(f'  engine counters (incl. legacy actions on the seat): buys {st("buys"):.1f} sells {st("sells"):.1f} refreshes {st("refreshes"):.1f} merges {st("merges"):.1f} leaks {st("leaks"):.1f}')
    ck = defaultdict(float); pk = defaultdict(float)
    for _, s in S:
        for k, v in s['cards'].items(): ck[k] += v
        for k, v in s['personal'].items(): pk[k] += v
    print('  机变 picks per seat-match:', {k: round(v / n, 2) for k, v in sorted(ck.items())}, '| personal choices:', {k: round(v / n, 2) for k, v in sorted(pk.items())})
    bt = defaultdict(float); bp = defaultdict(lambda: defaultdict(float))
    for _, s in S:
        for k in ('cardBattles', 'clean', 'units', 'leaked'): bt[k] += s['bounty'][k]
        for p, v in s['bounty']['byPayout'].items():
            for k, x in v.items(): bp[p][k] += x
    if bt['cardBattles']:
        print(f'  bounty (normal-round battles): card-battles {bt["cardBattles"] / n:.2f}/seat-match, clean (no bounty unit leaked) {bt["clean"] / bt["cardBattles"]:.3f},'
              f' units killed {1 - bt["leaked"] / max(1, bt["units"]):.3f} ({bt["units"] / n:.1f} units/seat-match)')
        for p, v in bp.items(): print(f'    payout {p}: n {int(v["n"])} clean {v["clean"] / max(1, v["n"]):.3f} units killed {1 - v["leaked"] / max(1, v["units"]):.3f} perfect round {v["perfect"] / max(1, v["n"]):.3f}')
    it = defaultdict(float); bi = defaultdict(float); sn = defaultdict(float)
    for _, s in S:
        for k, v in s['items'].items(): it[k] += v
        for k in ('n', 'equipped', 'onBoardAtBattle', 'neverEquippedLoose', 'neverEquippedGone'): bi[k] += s['boughtItems'][k]
        for k, v in s['snap'].items(): sn[k] += v
    print(f'  items per seat-match: gained by buy/reward {it["gained"] / n:.2f}, bought {it["bought"] / n:.2f} | legacy equips {it["equipCalls"] / n:.2f}'
          f' (on board units {it["equipBoard"] / n:.2f}, bench units {it["equipBench"] / n:.2f}, consume-on-equip {it["consumed"] / n:.2f})')
    if bi['n']:
        print(f'  bought items: {int(bi["n"])} | equipped by legacy {bi["equipped"] / bi["n"]:.3f}, on a board unit at a battle {bi["onBoardAtBattle"] / bi["n"]:.3f},'
              f' never equipped: still loose at the last battle {bi["neverEquippedLoose"] / bi["n"]:.3f}, gone (merged / destroyed / sold) {bi["neverEquippedGone"] / bi["n"]:.3f}')
    if sn['battles']:
        print(f'  per battle: items on board units {sn["board"] / sn["battles"]:.2f}, on bench units {sn["bench"] / sn["battles"]:.2f}, loose {sn["loose"] / sn["battles"]:.2f}'
              f' (equipment {sn["looseEquip"] / sn["battles"]:.2f})')
    last = [s['last'] for _, s in S if s.get('last')]
    if last: print(f'  last battle: board {mean(x["board"] for x in last):.2f} bench {mean(x["bench"] for x in last):.2f} loose {mean(x["loose"] for x in last):.2f} (equipment {mean(x["looseEquip"] for x in last):.2f})')
    rb = defaultdict(lambda: defaultdict(float)); rn = defaultdict(int)
    for _, s in S:
        for r, v in s['byRound'].items():
            for k, x in v.items(): rb[int(r)][k] += x
    for _, s in S:
        for r, *_ in s['rounds']: rn[r] += 1
    print('  per round (per seat alive at that settle): refresh', {r: round(rb[r]['refresh'] / max(1, rn[r]), 1) for r in sorted(rn)},
          '| level', {r: round(rb[r]['level'] / max(1, rn[r]), 2) for r in sorted(rn)}, '| item buys', {r: round(rb[r]['buyItem'] / max(1, rn[r]), 2) for r in sorted(rn)})
    # leaks: rounds with own bounty units vs without, by round
    br = defaultdict(lambda: [[], []])
    for _, s in S:
        for r, bu, bl, cnt in s['rounds']: br[r][1 if bu else 0].append((cnt, bl, bu))
    rows = []
    for r in sorted(br):
        no, yes = br[r]
        if len(yes) >= 5 and len(no) >= 5: rows.append((r, len(no), mean(x[0] for x in no), len(yes), mean(x[0] for x in yes), mean(x[1] for x in yes), mean(x[2] for x in yes)))
    if rows:
        print('  counted leaks by round, seats without / with own bounty units: ' + '; '.join(f'R{r} {a:.2f} (n {na}) / {b:.2f} (n {nb}, of which bounty units {bl:.2f} of {bu:.1f})' for r, na, a, nb, b, bl, bu in rows))
        w = sum(min(na, nb) for r, na, a, nb, b, bl, bu in rows)
        print(f'  weighted extra leaks in bounty rounds: {sum(min(na, nb) * (b - a) for r, na, a, nb, b, bl, bu in rows) / max(1, w):+.2f} per round (bounty units leaked {sum(min(na, nb) * bl for r, na, a, nb, b, bl, bu in rows) / max(1, w):.2f})')

def ab(cfg, on, off):
    A = {m['seed']: m for m in on['perMatch']}; B = {m['seed']: m for m in off['perMatch']}
    ks = sorted(set(A) & set(B)); dw = np.array([A[k]['win'] - B[k]['win'] for k in ks]); df = np.array([A[k]['faShare'] - B[k]['faShare'] for k in ks])
    sw = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    print(f'== A/B {cfg}: n {len(ks)} win {np.mean([B[k]["win"] for k in ks]):.3f} -> {np.mean([A[k]["win"] for k in ks]):.3f}  dwin {dw.mean():+.3f} ± {sw(dw):.3f}'
          f' (+{int((dw > 0).sum())}/-{int((dw < 0).sum())})  dfa {df.mean():+.3f} ± {sw(df):.3f}')
    bf = [m.get('bf') or {} for m in on['perMatch']]
    print(f'   bounty-aware layout swaps per match {mean(x.get("swaps", 0) for x in bf):.1f}, seat-preps with bounties {mean(x.get("preps", 0) for x in bf):.2f}')

if __name__ == '__main__':
    R = load(sys.argv[1])
    for cfg, c in R.items(): report(cfg, c)
    if '--vs' in sys.argv:
        O = load(sys.argv[sys.argv.index('--vs') + 1])
        for cfg in R:
            if cfg in O: ab(cfg, R[cfg], O[cfg])
