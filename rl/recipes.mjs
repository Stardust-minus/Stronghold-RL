// tools/rl/recipes.mjs — recipe bots (track 3 / m4): a recipe as the legacy bot's advisor plan (server/match/aiPlan.js,
// the executor the LLM plan seats use, §35.63), and the per-seat engine check used for tracking and snapshot curricula.
// Every rule is [ASSUMED] advisory, from docs/AI-STRATEGY.md §9 / §35.62–§35.63 and server/ai/strategy-community.js.
import { validPlan } from '../../server/match/aiPlan.js';

const C = (n) => `chess_char_${n}_a`;
// stage buy lists of the two detailed community builds (normal forms; the elite forms are kept)
const STAGED = {
  'kjerag-freeze': { // §9 freeze play: R1-3 cheap distinct members + collector + cold; R4-7 L3-4 engine; R8+ L5 merges, 谢讯
    early: [C('1_02'), C('3_14'), C('3_20'), C('3_11'), C('3_21')], // 角峰 初雪 耶拉 雪猎 空弦
    mid: [C('3_14'), C('4_22'), C('4_13'), C('3_20'), C('4_25'), C('3_21'), C('3_11'), C('1_02')], // + 银灰 灵知 魔王
    late: [C('5_14'), C('5_22'), C('4_13'), C('3_14'), C('4_22'), C('4_25'), C('3_20'), C('3_21'), C('3_11')], // + 凛御银灰 妮芙
    subs: { early: [], mid: [['swiftShip', 1]], late: [['swiftShip', 3], ['skillfulShip', 2]] },
    items: ['chess_item_5_02_e_a', 'chess_item_5_02_e_b'], // 谢拉格不融冰
  },
  'kazimierz-redeploy': { // §9 redeployment play: R1-3 cheap members/blockers; R4-7 远牙 + 坚守; R8+ 突袭/迅捷, 卡突袭迅
    early: [C('1_19'), C('2_18'), C('2_12'), C('3_12')], // 野鬃 灰毫 砾 瑕光
    mid: [C('4_20'), C('2_12'), C('3_12'), C('4_19'), C('1_19'), C('2_18')], // + 远牙 焰尾
    late: [C('4_20'), C('2_12'), C('3_12'), C('6_17'), C('5_10'), C('4_19'), C('1_19'), C('2_18')], // + 耀骑士临光 铃兰
    subs: { early: [], mid: [['steadShip', 3]], late: [['raidShip', 3], ['swiftShip', 2]] },
    items: [],
  },
};
const stageOf = (round) => (round <= 3 ? 'early' : round <= 7 ? 'mid' : 'late');

/** The advisor plan of seat `ps` following recipe `rc` this round (cached per round). */
export function recipePlan(m, ps, rc) {
  const c = ps._recipePlan;
  if (c && c.round === m.round && c.id === rc.id) return c.plan;
  const gd = m.gd, stage = stageOf(m.round || 1), st = STAGED[rc.id];
  let level = 1; for (const [r, l] of rc.rhythm || []) if ((m.round || 0) >= r) level = Math.max(level, l);
  const units = (st ? st[stage] : (rc.engines || [])).filter((id) => gd.chess(id));
  const allUnits = (rc.engines || []).filter((id) => gd.chess(id));
  const elite = allUnits.map((id) => gd.chess(id)?.goldenId).filter((id) => id && gd.chess(id));
  const items = (st ? st.items : (rc.engines || [])).filter((id) => !gd.chess(id) && gd.item?.(id));
  const subs = st ? st.subs[stage] : (m.round >= 8 ? (rc.subs || []).slice(0, 2).map((id) => [id, 2]) : []);
  const bonds = [{ id: rc.bond, priority: 5 }, ...subs.filter(([id]) => gd.bond(id)).map(([id, p]) => ({ id, priority: p }))];
  const plan = {
    bonds, buy: [...new Set(units)].slice(0, 32), keep: [...new Set([...allUnits, ...elite])].slice(0, 32),
    // the level rhythm 3-5-7-9 + L6 at R11 (§35.62 / §35.63) as a floor the executor levels to; freeze a missing engine piece
    economy: { level: Math.min(level, gd.maxShopLevel || 6), freezeSearch: true },
    ...(items.length ? { items: { buy: items.slice(0, 8), keep: items.slice(0, 32) } } : {}),
  };
  if (!validPlan(plan)) throw new Error(`recipe ${rc.id}: invalid advisor plan`);
  ps._recipePlan = { round: m.round, id: rc.id, plan };
  return plan;
}

const base = (gd, p) => gd.baseIdOf(p.id);
/** Seat state at a prep end: Σ layers, the recipe bond [tier, layers, count], and whether its engine is set up. */
export function engineState(m, ps, rc) {
  const gd = m.gd, bonds = ps.privateView().bonds || [];
  let sigma = 0; for (const b of bonds) if (b.active) sigma += b.layers || 0;
  const out = { sigma };
  if (!rc) return out;
  const b = bonds.find((x) => x.bondId === rc.bond);
  out.bond = [b?.tier || 0, b?.layers || 0, b?.count || 0];
  const onBoard = new Set([...ps.board.values()].filter(Boolean).map((p) => base(gd, p)));
  const has = (n) => onBoard.has(C(n));
  if (rc.id === 'kjerag-freeze') { // collector (初雪 / 银灰) + cold source (灵知 / 耶拉) with three active Kjerag; 凛御银灰 on board
    out.engine = !!b?.active && (b.count || 0) >= 3 && (has('3_14') || has('4_22')) && (has('4_13') || has('3_20'));
    out.alter = has('5_14');
  } else if (rc.id === 'kazimierz-redeploy') { // 远牙 + a receiver (砾 / 瑕光) with three active Kazimierz
    out.engine = !!b?.active && (b.count || 0) >= 3 && has('4_20') && (has('2_12') || has('3_12'));
  } else out.engine = !!b?.active && (b.count || 0) >= 4;
  return out;
}
