// Track M step 1: full matches with scripted route bots (routebots.mjs) — solo or co-op, matched seeds.
// usage: node routematch.mjs --config solo:ABYSS --seat0 nat|nat:<bond>|class|legacy [--mates legacy] --seeds 480 --seed0 7700000
//        --rehearsal 0 --boss-rehearsal 3 --out x.jsonl [--from i --to j]
import { appendFileSync } from 'node:fs';
import { Match } from '../../../server/match/Match.js';
import { VirtualScheduler } from '../../../server/match/scheduler.js';
import { getData } from '../../../server/data.js';
import { defineBotStrategy } from '../../../server/match/botStrategyContract.js';
import { runMatch } from '../../ai/run-match.mjs';
import * as RB from './routebots.mjs';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const data = getData({ log: { warn() {}, error() {}, info() {} } });
const [modeS, diff] = arg('config', 'solo:ABYSS').split(':');
const solo = modeS === 'solo', players = solo ? 1 : Number(modeS.replace('coop', '')) || 4;
const seat0 = arg('seat0', 'legacy'), mates = arg('mates', 'legacy'), allSame = arg('all', '') === '1';
const R0 = Number(arg('rehearsal', 0)), RB3 = Number(arg('boss-rehearsal', 3));
const seeds = Number(arg('seeds', 480)), seed0 = Number(arg('seed0', 7700000)), from = Number(arg('from', 0)), to = Number(arg('to', seeds));
const out = arg('out');
const ZB = ['yanShip', 'sargonShip', 'victoriaShip', 'kjeragShip', 'lateranoShip', 'egirShip', 'siracusaShip', 'kazimierzShip'];

async function one(seed) {
  const sched = new VirtualScheduler();
  const seats = [];
  for (let i = 0; i < players; i++) seats.push({ seat: i, playerId: `ai_${i}`, name: `AI-${i + 1}`, isBot: true, connected: true });
  const kinds = {}; const RS = Number(arg('route-seats', 1)); for (let i = 0; i < players; i++) kinds[`ai_${i}`] = i < RS || allSame ? seat0 : mates;
  const trace = {}; // pid -> round -> row (at each prep end, after the bot's own prepEnd)
  const onReady = (m, ps) => {
    const gd = m.gd, act = Object.entries(ps.bonds).filter(([, b]) => b.active);
    const nat = act.filter(([id]) => gd.bond(id)?.isCore).sort((a, b) => (b[1].count || 0) - (a[1].count || 0) || (b[1].layers || 0) - (a[1].layers || 0))[0];
    const P = ps._nat?.P || null, S = ps._nat?.S || null;
    const elites = [...ps.board.values()].filter((p) => p.kind === 'chess' && gd.isGolden(p.id)).length;
    (trace[ps.playerId] ||= {})[m.round] = {
      lv: ps.shop.level, lp: ps.lp, funds: ps.funds, el: elites, units: ps.deployCount,
      sig: act.reduce((s, [, b]) => s + (b.layers || 0), 0),
      top: nat ? [nat[0], nat[1].count || 0, nat[1].tier || 0, nat[1].layers || 0] : null,
      P: P ? [ps.bonds[P]?.count || 0, ps.bonds[P]?.tier || 0, ps.layers[P] || 0, RB.natDistinct(gd, ps)] : null,
      S: S ? [ps.bonds[S]?.count || 0, ps.bonds[S]?.tier || 0, ps.layers[S] || 0] : null,
      act: act.map(([id, b]) => [id, b.count || 0, b.tier || 0, b.layers || 0]),
    };
  };
  const setRehearsal = (m) => { m.botRehearsal = (m.round === m.gd.bossRound || m.round === m.gd.hiddenRound) ? RB3 : R0; };
  const oK = Number(arg('oracle', 0));
  const strat = defineBotStrategy(RB.seatStrategy(kinds, { onReady, setRehearsal, oracle: oK ? { K: oK, seed: arg('oracle-seed', 'fixed'), rounds: arg('oracle-rounds', 'all') } : null, oracleSeats: arg('oracle-seats', 'ai_0').split(',') }));
  let summary = null;
  const m = new Match({ roomCode: 'TRKM', mode: solo ? 'solo' : 'coop', difficulty: diff, seats, seed, data,
    log: { info() {}, debug() {}, warn() {}, error() {} }, scheduler: sched, botRehearsal: R0, botStrategy: strat,
    send: () => true, broadcast: () => {}, onEnd: (s) => { summary = s; } });
  let faPool = null;
  const startFA = m.startFinalAssault.bind(m);
  m.startFinalAssault = function (hidden) { const r = startFA(hidden); if (!hidden && !faPool) faPool = this.bossPool; return r; };
  const off = new Set(m.disabledBonds || []);
  m.start();
  await runMatch(m, () => summary != null, { maxSteps: 5e6 });
  const fa = faPool && faPool.maxHp > 0 ? Math.min(1, Math.max(0, 1 - Math.max(0, faPool.hp) / faPool.maxHp)) : 0;
  const rows = [];
  for (const ps of m.order) {
    const p = (summary?.players || []).find((x) => x.playerId === ps.playerId) || {};
    const t = trace[ps.playerId] || {};
    const rs = Object.keys(t).map(Number).sort((a, b) => a - b);
    const t2r = ps._nat ? rs.find((r) => t[r].P && t[r].P[1] >= 2) ?? null : null;
    const t2d = ps._nat ? rs.find((r) => t[r].P && t[r].P[3] >= ps._nat.T2) ?? null : null; // owned distinct (board + bench)
    rows.push({ seed, cfg: arg('config'), pid: ps.playerId, kind: kinds[ps.playerId], band: ps.bandId, win: p.victory ? 1 : 0, teamWin: (summary?.players || []).some((x) => x.victory) ? 1 : 0,
      round: p.eliminatedRound ?? m.round, lp: p.lp ?? ps.lp, fa, P: ps._nat?.P || null, S: ps._nat?.S || null, forcedLive: ps._nat?.forcedLive ?? null,
      off: ZB.filter((b) => off.has(b)), t2Round: t2r, t2DistinctRound: t2d, trace: t, errors: m.errorCount });
  }
  m.dispose();
  return rows;
}
for (let i = from; i < to; i++) {
  const rows = await one(seed0 + i);
  if (out) appendFileSync(out, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
}
