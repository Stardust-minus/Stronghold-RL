// R2 hand check (11-12): teacher-mode matches with reward version 2; per RL seat, each paid component against the
// engine's own counters (ps.stats leaks / kills, LP, the FA pool, the match summary). node r2check.mjs <configs> <matches/config>
import { SPEnv } from './env.mjs';
const cfgs = (process.argv[2] || 'solo:ABYSS,coop4ai:ABYSS,coop4:HARD').split(','), N = Number(process.argv[3] || 6);
const L = 28, W = 0.75;
const out = [];
const P = SPEnv.prototype, seat0 = P._seat, open0 = P._open, fin0 = P._finish;
P._seat = function (slot, ps) {
  const had = slot.seat.has(ps.playerId), s = seat0.call(this, slot, ps);
  if (!had) { const t = this._team(slot.m); s.chk = { lp0: Math.max(0, ps.lp || 0), team0: t.lp, n: t.n }; }
  return s;
};
P._open = function (i) {
  const slot = open0.call(this, i);
  if (slot) { // outer settle wrapper: the normal-combat kills of each round (unite kills = stats.kills − these)
    const m = slot.m, inner = m.settle; slot.normalKills = new Map();
    slot.uniteKills = new Map();
    m.settle = (plan, ur) => {
      const live = m.phase === 'COMBAT' || m.phase === 'UNITE', alive = live ? m.alivePlayers() : [];
      const before = new Map(alive.map((ps) => [ps.playerId, [ps.stats.kills, Number(m.lastResults.get(ps.playerId)?.killed) || 0]]));
      const ret = inner(plan, ur);
      for (const ps of alive) { // the engine's own settle increment of stats.kills minus the normal-combat kills = its 联防 kills
        const [k0, normal] = before.get(ps.playerId);
        slot.normalKills.set(ps.playerId, (slot.normalKills.get(ps.playerId) || 0) + normal);
        slot.uniteKills.set(ps.playerId, (slot.uniteKills.get(ps.playerId) || 0) + Math.max(0, ps.stats.kills - k0 - normal));
      }
      return ret;
    };
  }
  return slot;
};
P._finish = function (slot) {
  const m = slot.m, t = this._team(m);
  const cap = [...m.players.values()].map((ps) => ({ pid: ps.playerId, leaks: ps.stats.leaks, kills: ps.stats.kills, lp: Math.max(0, ps.lp || 0),
    normalKills: slot.normalKills.get(ps.playerId) || 0, uniteKills: slot.uniteKills.get(ps.playerId) || 0, s: slot.seat.get(ps.playerId) }));
  const nTerms = this.terms.length;
  fin0.call(this, slot);
  const res = this.results.at(-1), terms = this.terms.slice(nTerms);
  res.check = { teamLp: t.lp, n: t.n, cap: cap.map(({ s, ...c }) => ({ ...c, chk: s?.chk, traj: s?.traj })), terms: terms.map((x) => ({ traj: x.traj, reward: x.reward })) };
};
for (const [ci, cfg] of cfgs.entries()) {
  const env = new SPEnv({ envs: 3, configs: [cfg], seed: 8_800_000 + ci * 10_000, envSeed: 88, maxMatches: N, humans: 'random', legacyMates: 0.3,
    mode: 'teacher', enforce: true, obsVersion: 3, rehearsal: 0, bossRehearsal: 3, reward: { round: 0.05, fa: 1.5 },
    rewardV: 2, r2: { layer: 0.002 }, shaping: { gamma: 0.99, wTier: 0.06, wStack: 0.3, uncapped: true } });
  env.reset();
  for (const res of env.results) {
    const cleared = res.seats.some((s) => s.win) || res.faShare >= 1 - 1e-9;
    for (const st of res.seats) {
      if (!st.r2) continue;
      const c = res.check.cap.find((x) => x.pid === st.playerId), term = res.check.terms.find((x) => x.traj === c.traj);
      const ev = st.r2.ev;
      const exp = {
        win: cleared ? 3 : 0, faTeam: 1.5 * res.faShare, faOwn: 0.5 * ev.ownFa, leak: -0.5 * c.leaks / L,
        save: 0.25 * c.uniteKills / L,
        lp: ((1 - W) * (c.lp - c.chk.lp0) + W * (res.check.teamLp - c.chk.team0) / res.check.n) / L,
        elim: ev.eliminated != null && !st.win ? -0.5 : 0, layer: 0.002 * ((ev.layerHW || 0) - (ev.layerHW0 || 0)) };
      const got = { win: st.r2.win, faTeam: st.r2.faTeam, faOwn: st.r2.faOwn, leak: st.r2.leak, save: st.r2.save, lp: st.r2.lp, elim: st.r2.elim, layer: st.r2.layer };
      const sum = Object.values(got).reduce((a, b) => a + b, 0) + (st.r2.shape || 0);
      out.push({ config: res.config, seed: res.seed, pid: st.playerId, win: st.win, cleared, faShare: +res.faShare.toFixed(3), endRound: st.survivalRound,
        counts: { leaksEnv: ev.leaks, leaksEngine: c.leaks, savesEnv: ev.saves, uniteKillsEngine: c.uniteKills, killsOutsideSettle: c.kills - c.normalKills - c.uniteKills, layerHW0: ev.layerHW0, layerHW: ev.layerHW, ownFa: +ev.ownFa.toFixed(3) },
        got, exp, maxAbsDiff: Math.max(...Object.keys(exp).map((k) => Math.abs(exp[k] - got[k]))), termReward: term?.reward, sumComponents: sum });
    }
  }
}
console.log(JSON.stringify(out));
