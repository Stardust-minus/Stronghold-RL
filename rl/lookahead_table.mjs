// Static tables for the bond look-ahead probe features: piece code -> bond indices; per bond: thresholds, core flag.
//   node tools/rl/lookahead_table.mjs <out.json>   (codes = spec.mjs vocabulary, version 2 and 3 alike)
import { writeFileSync } from 'node:fs';
import { getData } from '../../server/data.js';
import { Match } from '../../server/match/Match.js';
import { VirtualScheduler } from '../../server/match/scheduler.js';
import { Encoder } from './spec.mjs';
const data = getData({ log: { warn() {}, error() {}, info() {} } });
const m = new Match({ roomCode: 'LA', mode: 'solo', difficulty: 'NORMAL', seats: [{ seat: 0, playerId: 'p', name: 'p', isBot: true, connected: true }],
  seed: 1, data, scheduler: new VirtualScheduler(), log: { info() {}, warn() {}, error() {} }, send: () => true, broadcast: () => {}, onEnd: () => {} });
const gd = m.gd, enc = new Encoder(gd, { version: 2 }), V = enc.vocab;
const bonds = {};
for (const id of Object.keys(gd.raw.chess || {})) {
  const code = V.piece.get(gd.baseIdOf(id)); if (!code) continue;
  const set = new Set(bonds[code] || []); for (const b of gd.raw.chess[id].bonds || []) { const k = V.bond.get(b); if (k !== undefined) set.add(k); }
  bonds[code] = [...set].sort((a, b) => a - b);
}
const bondInfo = V.bondIds.map((b) => ({ id: b, thresholds: gd.bond(b)?.thresholds || [], core: !!gd.bond(b)?.isCore }));
writeFileSync(process.argv[2], JSON.stringify({ pieceBonds: bonds, bonds: bondInfo, pieceVocab: V.sizes.piece }));
console.log('pieces with bonds', Object.keys(bonds).length, 'bonds', bondInfo.length, 'max bonds per piece', Math.max(...Object.values(bonds).map((x) => x.length)));
m.dispose?.();
