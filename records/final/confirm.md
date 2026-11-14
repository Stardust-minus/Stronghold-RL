## Confirmation round (11-14): fresh seed set --seed 300000000, b protocol (code-A7, c9 pool + xboss-v3), 476 per cell

Fresh set: solo:ABYSS seeds 302,000,000-302,000,475, coop4h1:ABYSS 880,000,000-880,000,475. Untouched by any selection. The only other use: the JS check matches of the release exports (`rl/release/play.mjs`, parity only, chose nothing) on solo:ABYSS 302,000,000-302,000,007 and coop4h1:ABYSS 880,000,000-880,000,007; the last column drops them.

### 1. Fresh-seed headline (paired V5X-s1-90M - MLP-s2-70M)

| candidate | solo:ABYSS | coop4h1:ABYSS | mean | solo / coop w/o the check-match seeds |
|---|---|---|---|---|
| FIN-MLP-s2-70M | 0.769 | 0.731 | 0.750 | 0.765 / 0.729 |
| FIN-V5X-s1-90M | 0.790 | 0.744 | 0.767 | 0.791 / 0.746 |

Paired V5X - MLP: solo +0.021 ± 0.024 (n 476; w/o check-match seeds +0.026 ± 0.024), coop +0.013 ± 0.027 (n 476; w/o +0.017 ± 0.028), mean +0.017 ± 0.018

Stacking (fresh set; solo / coop rows):

| candidate | cfg | natMain / T2 | main tier | main / sec / Σ layers | elites | main bond |
|---|---|---|---|---|---|---|
| FIN-MLP-s2-70M | solo | 0.99 / 0.97 | 1.98 | 185 / 87 / 325 | 5.6 | 拉特兰 0.64, 维多利亚 0.24 |
| FIN-MLP-s2-70M | coop | 0.92 / 0.82 | 1.89 | 185 / 75 / 302 | 5.7 | 维多利亚 0.20, 拉特兰 0.20 |
| FIN-V5X-s1-90M | solo | 0.99 / 0.97 | 1.98 | 191 / 89 / 336 | 5.5 | 拉特兰 0.65, 维多利亚 0.24 |
| FIN-V5X-s1-90M | coop | 0.89 / 0.75 | 1.80 | 184 / 68 / 295 | 4.6 | 拉特兰 0.20, 维多利亚 0.20 |

### 2. Human-robustness cells (coop4h1:ABYSS, same fresh seeds; nation cells on live seeds)

| cell | live seeds | MLP-s2-70M | V5X-s1-90M | Δ V5X - MLP (paired) | MLP hole vs own legacy | V5X hole vs own legacy | MLP natMain / T2, Σ, elites | V5X natMain / T2, Σ, elites |
|---|---|---|---|---|---|---|---|---|
| 叙拉古 | 269 | 0.736 | 0.773 | +0.037 ± 0.030 | +0.022 ± 0.034 | +0.063 ± 0.033 | 0.98 / 0.89, 300, 5.7 | 0.95 / 0.84, 319, 5.0 |
| 炎 | 306 | 0.771 | 0.758 | -0.013 ± 0.030 | +0.075 ± 0.031 | +0.046 ± 0.030 | 0.95 / 0.85, 322, 5.9 | 0.92 / 0.81, 313, 4.9 |
| 谢拉格 | 302 | 0.652 | 0.619 | -0.033 ± 0.034 | -0.132 ± 0.032 | -0.175 ± 0.032 | 0.92 / 0.77, 278, 5.5 | 0.88 / 0.69, 253, 4.0 |
| 阿戈尔 | 301 | 0.704 | 0.681 | -0.023 ± 0.032 | -0.007 ± 0.032 | -0.010 ± 0.033 | 0.96 / 0.86, 285, 5.7 | 0.93 / 0.78, 285, 4.6 |
| 卡西米尔 | 298 | 0.661 | 0.681 | +0.020 ± 0.032 | -0.084 ± 0.032 | -0.077 ± 0.031 | 0.91 / 0.79, 290, 5.6 | 0.90 / 0.74, 277, 4.4 |
| 维多利亚 | 301 | 0.694 | 0.688 | -0.007 ± 0.033 | -0.023 ± 0.032 | -0.066 ± 0.033 | 0.91 / 0.77, 282, 5.5 | 0.93 / 0.70, 293, 4.6 |
| 拉特兰 | 304 | 0.645 | 0.645 | +0.000 ± 0.031 | -0.128 ± 0.032 | -0.141 ± 0.033 | 0.85 / 0.71, 290, 5.5 | 0.88 / 0.66, 284, 4.3 |
| 萨尔贡 | 299 | 0.722 | 0.766 | +0.043 ± 0.030 | +0.017 ± 0.029 | +0.027 ± 0.030 | 0.96 / 0.87, 300, 5.8 | 0.94 / 0.81, 309, 5.1 |
| class | all 476 | 0.674 | 0.704 | +0.029 ± 0.026 | -0.057 ± 0.025 | -0.040 ± 0.026 | 0.91 / 0.79, 293, 5.5 | 0.90 / 0.76, 301, 4.6 |

### 3. Real-player-weighted coop4h1 (real-player nation mix)

| candidate | real-player-weighted coop4h1 | legacy-proxy coop (fresh) |
|---|---|---|
| FIN-MLP-s2-70M | 0.702 ± 0.010 | 0.731 |
| FIN-V5X-s1-90M | 0.713 ± 0.010 | 0.744 |

Paired weighted Δ V5X - MLP = +0.012 ± 0.012
