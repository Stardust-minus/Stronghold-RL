// tools/rl/picker.mjs — boss-round candidate picker (track 2): a small head on the policy trunk's context `c` and one
// feature row per legacy candidate layout (bot.js layoutCandidatesForPicker; planLayout only, no simulation).
//   logit_j = w2 · relu(W1 [c, row_j] + b1) + b2 + prior · row_j[1]   (row_j[1] = the placed default)
import { readFileSync } from 'node:fs';

export const PICK_F = 24;
/** The feature row of candidate j (PICK_F floats). */
export function pickerRow(feat, variants, isDefault, f0) {
  const r = new Float32Array(PICK_F);
  r[0] = 1; r[1] = isDefault ? 1 : 0;
  for (const v of variants) if (v >= 0 && v < 8) r[2 + v] = 1;
  r[10] = variants.length / 8; r[11] = feat.value; r[12] = feat.value - f0.value; r[13] = feat.cov; r[14] = feat.covMin; r[15] = feat.flyCov;
  r[16] = feat.onRoad; r[17] = feat.leader; r[18] = feat.diff; r[19] = feat.units / 12; r[20] = feat.leaderCov; r[21] = feat.leaderBlock / 6;
  r[22] = feat.leaderExp / 3; r[23] = feat.healCov;
  return r;
}
/** Lexicographic rehearsal ranking (fewer leaks, more leader damage, more kills; the earlier on ties): best index. */
export function bestOf(scores) {
  let b = -1;
  for (let j = 0; j < scores.length; j++) { const s = scores[j], t = b >= 0 ? scores[b] : null; if (!s) continue;
    if (!t || s.leaks < t.leaks || (s.leaks === t.leaks && (s.boss > t.boss || (s.boss === t.boss && s.kills > t.kills)))) b = j; }
  return b;
}
export class PickerHead {
  constructor(j) { Object.assign(this, j); this.h = new Float64Array(j.H); }
  static load(path) { return new PickerHead(JSON.parse(readFileSync(path, 'utf8'))); }
  /** logits over the candidates (rows: Float32Array[]) given the trunk context c (Float32Array C). */
  score(c, rows) {
    const { C, F, H, W1, b1, w2, b2, prior } = this, out = new Float64Array(rows.length);
    const hc = new Float64Array(H);
    for (let k = 0; k < H; k++) { let s = b1[k]; const w = W1[k]; for (let i = 0; i < C; i++) s += w[i] * c[i]; hc[k] = s; }
    rows.forEach((r, j) => {
      let z = b2;
      for (let k = 0; k < H; k++) { let s = hc[k]; const w = W1[k]; for (let i = 0; i < F; i++) s += w[C + i] * r[i]; if (s > 0) z += w2[k] * s; }
      out[j] = z + prior * r[1];
    });
    return out;
  }
}
