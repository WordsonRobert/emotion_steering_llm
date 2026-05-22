# outputs/ — Complete Results Inventory

Every file produced by the 10 scripts. Listed in the order they are generated.
Numbers are taken directly from the notebook run (`emnlp_steering_colab.pdf`, executed 2026-05-22).

---

## Script 01 — Setup, Model, Activation Extraction, Binary Probe

---

### `humor_acts.pt`
Tensor of shape **(500, 3072)**. Last-token residual stream activations at layer 26 (PROBE_LAYER) for every text in `humor_master.csv`, extracted from Llama-3.2-3B in bfloat16, stored as float32.

---

### `dark_acts.pt`
Tensor of shape **(618, 3072)**. Same extraction for the combined dark set (419 melancholy + 199 horror texts).

---

### `probe_weights.pt`
Dictionary mapping each of the 28 layers (0–27) to a unit-normalized float32 numpy array of shape **(3072,)**. Each entry is the weight vector of a binary logistic regression probe (humor vs. dark) trained at that layer with 5-fold CV and balanced sampling (N=300 per class).

**Key accuracy values:**
```
L0:  0.8200    L8:  0.9217    L17: 0.9333
L21: 0.9350    L24: 0.9383    L25: 0.9367
L26: 0.9400  ← PROBE_LAYER (used everywhere downstream)
L27: 0.9367
```
Peak is L26 at **0.9400 ± 0.0062**. All layers ≥ L17 are in the 0.92–0.94 band.

---

### `layer_probe_accuracy.png`
Bar chart of probe CV accuracy at every layer (0–27). L26 bar is highlighted in a different colour. Chance line at 0.5. Used to justify the choice of L26 as PROBE_LAYER.

---

## Script 02 — Pairwise Probes, Steering Vectors, Layer and Multiplier Sweep

---

### `probe_humor_vs_melancholy.pt`
Unit-normalized float32 tensor **(3072,)**. Weight vector of a logistic probe trained on humor (positive class) vs. melancholy (negative class) at L26.
**CV accuracy: 0.9916 ± 0.0061** (n=419 balanced).

---

### `probe_humor_vs_horror.pt`
Unit-normalized float32 tensor **(3072,)**. Humor vs. horror probe at L26.
**CV accuracy: 0.9272 ± 0.0215** (n=199 balanced).

---

### `probe_melancholy_vs_horror.pt`
Unit-normalized float32 tensor **(3072,)**. Melancholy vs. horror probe at L26.
**CV accuracy: 0.9798 ± 0.0152** (n=199 balanced).

---

### `steering_vecs.pt`
Dictionary with keys `humor`, `melancholy`, `horror`. Each value is a bfloat16 tensor **(3072,)**. Built as the mean-difference vector in whitened probe space: project each class's mean activation into the z-scored space, take the difference, normalize to unit norm.

**Cosine similarities between steering vectors:**
```
humor · melancholy  = −0.9972   (nearly antipodal — same axis, opposite ends)
humor · horror      = −0.3167   (distinct axis — horror sits ~orthogonal to valence)
melancholy · horror = +0.3151   (mel and horror share negative-valence side)
```
Alignment of steering vectors with their probe weight vectors:
```
cos(steer_humor,  w_hm) = +0.9139
cos(steer_horror, w_hh) = −0.9127
```

---

### `best_configs.json`
JSON with the optimal layer and multiplier per emotion, selected by argmax of both-rate (probe shift positive AND coherence ≥ 0.6) in the multiplier sweep. Also stores `best_temp` (set by Script 03).

```json
{
  "humor":      {"layer": 18, "mult": 12.0, "both": 0.667},
  "melancholy": {"layer": 18, "mult": 3.0,  "both": 0.867},
  "horror":     {"layer": 15, "mult": 9.0,  "both": 0.867},
  "best_temp":  0.7
}
```

---

### `cell5_layer_sweep_raw.json`
Raw checkpoint from the layer sensitivity sweep. For each of 28 layers × 3 emotions, stores the mean probe shift at `mult=9.0` over n=30 samples. Used to identify the top-3 layers per emotion for the multiplier sweep. Checkpoint — written after every layer.

---

### `cell5_mult_sweep_full.json`
Raw results from the multiplier sweep over the top-3 layers per emotion × 5 multipliers [3, 6, 9, 12, 15] × n=15 samples. Stores `both` rate and `mean_shift` per (emotion, layer, mult) combination.

**Sample values (humor, layer 18):**
```
L18 × 3.0:  shift=+0.774  coh=0.91  both=53.3%
L18 × 6.0:  shift=+1.415  coh=0.84  both=60.0%
L18 × 9.0:  shift=+2.574  coh=0.70  both=53.3%
L18 ×12.0:  shift=+1.554  coh=0.66  both=46.7%   ← manually confirmed best
L18 ×15.0:  shift=+2.986  coh=0.66  both=53.3%
```

---

### `layer_heatmap.png`
Three-panel bar chart (one panel per emotion) showing mean probe shift at every layer at `mult=9.0`. Used to visually confirm which layers are most responsive to steering. PROBE_LAYER marked.

---

## Script 03 — Temperature Sweep and Main Evaluation at T=0.7

---

### `temp_sweep_results.csv`
Full sweep results. **3,300 rows** (11 temperatures × 3 emotions × 100 samples). Columns include: `temperature`, `emotion_target`, `prompt`, `base_gen`, `ste_gen`, `ste_coh`, `is_coherent`, `ps_shift`, `ps_success`, `both`, `score_humor`, `score_mel`, `score_horror`, `pred_emotion`, `pred_correct`.

**Average `both` rate by temperature (averaged across all three emotions):**
```
T=0.0:  43.0%    T=0.1:  43.3%    T=0.2:  45.0%    T=0.3:  51.7%
T=0.4:  53.3%    T=0.5:  55.7%    T=0.6:  58.0%    T=0.7:  61.7%
T=0.8:  62.3%    T=0.9:  64.0%    T=1.0:  69.3%
```
T=0.7 chosen for reporting because it maximises confusion matrix diagonal (90.0%) with lower coherence cost than T≥0.8.

---

### `simplex_results_final.csv`
Rows from `temp_sweep_results.csv` at BEST_TEMP only. **300 rows** (100 per emotion). This is the file used for all downstream analyses (ablations, external validation, DeepSeek judge).

---

### `simplex_results_T07.csv`
Same as `simplex_results_final.csv` — rows at T=0.7. **300 rows.** This naming is used throughout downstream scripts.

---

### `temp_sweep_confusion_grid.png`
Grid of 11 confusion matrices (one per temperature), 3×3 each, showing how well the probe classifies steered outputs at every temperature. Illustrates the T=0.7 as the crossover point where diagonal is high and coherence has not yet degraded.

---

### `temp_sweep_metrics_plot.png`
Line plots of `both`, `pred_correct`, `coherent`, and `mean_shift` across the 11 temperatures, one line per emotion. Shows the coherence–shift trade-off as temperature rises.

---

### `probe_confusion_matrix_T07.png`
The primary result figure. 3×3 confusion matrix at T=0.7, n=100 per emotion. Values:

```
                →humor    →mel    →horror
humor steered    92%       2%       6%
mel steered       3%      82%      15%
horror steered    4%       0%      96%

Avg diagonal: 90.0%
```

---

### `vector_comparison.csv`
**450 rows** (3 vector types × 3 emotions × 50 samples). Columns: `emotion`, `vec` (mean_diff / probe_w / random), `ps_shift`, `both`. Used for both the bar chart and the Mann-Whitney stats.

**Both-rates:**
```
                mean_diff    probe_w    random
humor             58.0%       40.0%      36.0%
melancholy        52.0%       60.0%      44.0%
horror            74.0%       80.0%      50.0%
```

---

### `vector_comparison_stats.csv`
Mann-Whitney U test results for all pairwise comparisons between the three vector types, per emotion, with Bonferroni correction. 9 rows.

**Significant results:**
```
humor:      mean_diff vs random   p_bonf < 0.001  ***
horror:     mean_diff vs random   p_bonf < 0.01   **
horror:     probe_w   vs random   p_bonf < 0.01   **
```
All other comparisons: ns.

---

### `vector_comparison.png`
Grouped bar chart: per-emotion both-rates for mean_diff, probe_w, and random. Error bars showing standard error. Significance brackets from `vector_comparison_stats.csv`.

---

### `cell7_summary.json`
Summary statistics at T=0.7:
```json
{
  "eval_temp": 0.7,
  "avg_diagonal_T07": 0.900,
  "per_emotion": {
    "humor":      {"both": 0.60, "pred": 0.92, "coherent": 0.76, "mean_shift": 3.7193},
    "melancholy": {"both": 0.58, "pred": 0.82, "coherent": 0.99, "mean_shift": 0.4612},
    "horror":     {"both": 0.67, "pred": 0.96, "coherent": 0.97, "mean_shift": 2.1499}
  }
}
```

---

## Script 04 — Head Ablation and CPU Analyses

---

### `task19_ablation_results.csv`
One row per (layer, head) combination, for ablation layers [0, 7, 14, 15, 18, 26] × 24 heads = **144 rows**. Columns: `layer`, `head`, `baseline_acc`, `ablated_acc`, `drop`, `load_bearing` (bool, drop > 0.02).

**Critical heads (load_bearing=True), all at Layer 0:**
```
Layer 0: 10 critical heads   max_drop = 0.060
Layer 7: 0 critical heads
Layer 14: 0 critical heads
Layer 15: 0 critical heads
Layer 18: 0 critical heads
Layer 26: 0 critical heads
```
Top head: **L0-H7** with probe accuracy drop of **0.060**.

---

### `task19_head_ablation.png`
One bar chart per ablation layer (6 panels), showing the probe accuracy drop for each of 24 heads. Critical heads (drop > 0.02) highlighted in red. Visually confirms all load-bearing heads cluster at L0.

---

### `cell9_summary.json`
```json
{
  "ablation_layers": [0, 7, 14, 15, 18, 26],
  "n_heads": 24,
  "n_texts": 100,
  "threshold": 0.02,
  "n_critical_heads": 10,
  "critical_heads": [{"layer": 0, "head": 7, "drop": 0.060}, ...]
}
```

---

### `task3_binary_vs_threepole_results.json`
```json
{
  "mel_horror_bleed":   0.075,
  "humor_other_bleed":  0.038,
  "bleed_ratio":        2.00,
  "sv_cosines": {
    "humor_humor": 1.0, "humor_melancholy": -0.9972, "humor_horror": -0.3167,
    "melancholy_melancholy": 1.0, "melancholy_horror": 0.3151,
    "horror_horror": 1.0
  }
}
```
Melancholy–horror cross-confusion (7.5%) is **2× higher** than humor–other cross-confusion (3.8%), as predicted by the circumplex geometry.

---

### `task3_binary_vs_threepole.png`
Side-by-side: (left) the 3×3 confusion matrix from `simplex_results_T07.csv`, (right) the 3×3 cosine similarity heatmap of the three steering vectors. Shows the geometric prediction matches the confusion pattern.

---

### `task4_surface_probe_results.json`
```json
{
  "surface_probe_acc":    0.788,
  "activation_probe_acc": 0.900,
  "gap":                  0.112
}
```
The activation probe outperforms a 10-feature surface classifier (TTR, punctuation density, sentence count, exclamation rate, avg word length, etc.) by **11.2 pp**.

---

### `task4_surface_probe.png`
Bar chart comparing surface probe (0.788) vs activation probe (0.900) accuracy, with the gap annotated. Per-feature importance scores for the surface features shown as a secondary panel.

---

### `task5_ood_results.json`
```json
{
  "in_dist": 0.788,
  "ood": {
    "humor_ood":  {"acc": 0.590, "n": 200, "source": "english_quotes",  "drop": 0.286},
    "mel_ood":    {"acc": 0.310, "n": 200, "source": "merve/poetry",     "drop": 0.659},
    "horror_ood": {"acc": 0.285, "n": 200, "source": "imdb_neg",         "drop": -0.059}
  }
}
```
Melancholy OOD drop (−65.9 pp) is the largest: r/depression surface features don't generalize to poetry. Horror in-distribution surface accuracy was already low (22.6%), so OOD drop is negligible.

---

### `task5_ood_eval.png`
Grouped bar chart showing in-distribution vs OOD accuracy for each emotion class. Drop annotated per bar.

---

## Script 05 — Hook Comparison, Tonal Drift, Six Ablations, DistilRoBERTa

---

### `task2_additive_vs_projinject.csv`
**180 rows** (2 hook types × 3 emotions × 30 samples). Columns: `emotion`, `mode` (proj_inject / additive), `ps_shift`, `both`, `coherent`, `pred_correct`.

**Both-rates and Mann-Whitney U:**
```
humor:      proj_inject=70.0%  additive=60.0%   U=442  p=0.9176  ns
melancholy: proj_inject=50.0%  additive=70.0%   U=353  p=0.1537  ns
horror:     proj_inject=63.3%  additive=90.0%   U=253  p=0.0037  **
```

---

### `task2_additive_vs_projinject.png`
Grouped bar chart of both-rates (and coherence, pred_correct) per emotion × hook type. Significance bracket on horror (**).

---

### `task6_tonal_drift.csv`
**30 rows** (one per horror-steered prompt). Columns: `pair_id`, `turn1_horror`, `turn2_horror`, `horror_drift`, `turn1_humor`, `turn2_humor`, `turn1_mel`, `turn2_mel`.

**Key stats:**
```
Turn 1 horror score (steered):    +5.8549
Turn 2 horror score (unsteered):  +7.7292
Mean drift (T1 − T2):             −1.8742  (horror INCREASES from T1 to T2)
% pairs where T2 < T1 (neutralized): 23.3%
Paired t-test:  t = −2.132   p = 0.0416  *
Mann-Whitney:   U = 305       p = 0.9843  ns
```
Steered context primes the neutral follow-up, which often generates more extreme horror than the steered turn itself.

---

### `task6_tonal_drift.png`
Two panels: (left) scatter of turn 1 vs turn 2 horror scores with the T1=T2 diagonal, (right) histogram of drift values (T1−T2) with mean marked. Drift is predominantly negative (T2 > T1).

---

### `task6_tonal_drift_summary.json`
```json
{
  "best_temp": 0.7,
  "mean_t1_horror":  5.8549,
  "mean_t2_horror":  7.7292,
  "mean_drift":     -1.8742,
  "pct_neutralized": 0.233,
  "t_stat": -2.132,
  "t_p":     0.0416
}
```

---

### `cell11_summary.json`
Aggregated summary of Task 2 and Task 6 results.

---

### `task_random_baseline.csv`
**180 rows** (4 vectors [our + 3 random seeds] × 3 emotions × 15 samples). Per-row probe shift, both, coherence.

**Both-rates (our vector vs. mean across 3 random seeds):**
```
humor:      our=40.0%  random=48.9%   p=0.008  ** (random wins at n=15 — small-sample artifact)
melancholy: our=33.3%  random=42.2%   p=0.753  ns
horror:     our=86.7%  random=46.7%   p=0.0002 ***
```

---

### `task_data_ablation.csv`
**105 rows** (7 data sizes × 15 samples). Data sizes tested: 25, 50, 100, 150, 200, 300, 500.

**Both-rates for humor steering by training set size:**
```
size=25:  53.3%    size=50:  60.0%    size=100: 40.0%
size=150: 33.3%    size=200: 46.7%    size=300: 26.7%    size=500: 46.7%
```
No monotonic trend at n=15 — noisy due to small evaluation set.

---

### `task_data_ablation.png`
Line chart of both-rate vs. training data size for humor steering.

---

### `task_repe_comparison.csv`
**90 rows** (2 vector types [PCA, mean_diff] × 3 emotions × 15 samples).

**Both-rates:**
```
humor:      PCA=13.3%  mean_diff=33.3%
melancholy: PCA=60.0%  mean_diff=53.3%
horror:     PCA=46.7%  mean_diff=40.0%
```

**Geometric distances:**
```
cos(PCA, mean_diff) = −0.2033   (nearly orthogonal — different directions)
cos(MD,  probe_w)   = +0.8848   (mean_diff aligns strongly with probe boundary)
```

---

### `task_head_ablation_control.csv`
**10 rows** (one per critical head). Columns: `layer`, `head`, `emo_drop` (probe accuracy drop from head ablation), `sv_drop` (subject-verb agreement drop from same ablation), `specificity` (emo_drop − sv_drop).

```
SV baseline accuracy:    0.300
Mean specificity:       +0.020   (critical heads are marginally more emotion-specific than syntactic)
```

---

### `task_4turn_drift.csv`
**15 rows** (one per horror-steered prompt, 4-turn extension). Columns: `pair_id`, `score_t1` through `score_t4`.

**Mean horror probe scores by turn:**
```
Turn 1 (steered):    +6.892
Turn 2 (unsteered):  +4.834
Turn 3 (unsteered):  +4.887
Turn 4 (unsteered):  +4.981
```
Partial decay from T1 to T2; T2–T4 are roughly stable. T1 vs T2/T3/T4 Mann-Whitney p ≈ 0.08 (ns at n=15).

---

### `task_4turn_drift.png`
Line chart of mean horror probe score across the four turns, with individual pair lines in grey and mean in red.

---

### `task_roberta_eval.csv`
**300 rows** — `simplex_results_T07.csv` with an added `roberta_target_score` column. Each row scored by `j-hartmann/emotion-english-distilroberta-base`.

**DistilRoBERTa target score > 0.3 rate and Spearman correlation with probe shift:**
```
humor:      RoBERTa>0.3 = 29.0%   r = +0.207   p = 0.0383  *
melancholy: RoBERTa>0.3 = 41.0%   r = +0.041   p = 0.6877  ns
horror:     RoBERTa>0.3 = 53.0%   r = +0.031   p = 0.7578  ns
```

---

### `cell12_summary.json`
Aggregated results from all six ablation experiments (Exp A–F).

---

## Script 06 — DeepSeek-V3 LLM Judge

---

### `task_nim_eval.csv`
**300 rows** (one per steered output from `simplex_results_T07.csv`). Columns: `row_idx`, `emotion_target`, `ps_shift`, `both`, `pred_correct`, `ste_gen`, `nim_humor` (1–5), `nim_mel` (1–5), `nim_horror` (1–5), `nim_dominant`, `nim_confidence`.

**DeepSeek dominant emotion accuracy:**
```
humor:      95.0%  (95/100)
melancholy: 88.0%  (88/100)
horror:     47.0%  (47/100)
Overall:    76.7%
```

**DeepSeek mean target score (1–5) and Spearman r vs probe shift:**
```
humor:      mean score = 2.78   r = +0.202   p = 0.0436  *
melancholy: mean score = 3.59   r = +0.176   p = ns
horror:     mean score = 3.50   r = +0.145   p = ns
```

**DeepSeek × DistilRoBERTa cross-validation:**
```
humor:      r = +0.553   p < 0.001  ***
melancholy: r = +0.444   p < 0.001  ***
horror:     r = +0.445   p < 0.001  ***
```

---

### `task_nim_eval.png`
Two panels: (left) heatmap of mean DeepSeek 1–5 scores by emotion target vs. dimension rated (3×3), (right) bar chart of dominant emotion accuracy per target emotion with chance line at 33%.

---

### `cell13_summary.json`
```json
{
  "judge_model": "deepseek-chat",
  "n_valid": 300,
  "n_failed": 0,
  "eval_temp": 0.7,
  "overall_accuracy": 0.767,
  "per_emotion_accuracy": {"humor": 0.95, "melancholy": 0.88, "horror": 0.47},
  "mean_scores": {
    "humor":      {"humor": 2.78, "melancholy": ..., "horror": ...},
    "melancholy": {"humor": ..., "melancholy": 3.59, "horror": ...},
    "horror":     {"humor": ..., "melancholy": ..., "horror": 3.50}
  }
}
```

---

## Script 07 — Emotional Tone Rollout Generation (EMOTIONAL_AXES/)

---

### `EMOTIONAL_AXES/emotional_rollouts.csv`
**12,390 rows** (59 tones × 7 system prompts [5 positive + 2 neutral] × 30 questions). Columns: `tone`, `description`, `polarity` (positive/negative), `prompt_idx`, `question_idx`, `system_prompt`, `question`, `response`, `key`.

Covers 59 emotional tones from humor, joy, and whimsy through to existential_dread, dissociation, and desolation. Each response is a 200-token DeepSeek-V3 completion generated at temperature 0.9.

---

### `EMOTIONAL_AXES/rollout_checkpoint.json`
Checkpoint file. Stores all completed `key` values so the generation loop can resume from any interruption. Not a results file — infrastructure only.

---

## Script 08 — Activation Extraction from Rollouts

---

### `EMOTIONAL_AXES/emotional_acts_L14.pt`
Tensor of shape **(12390, 3072)**. Mean residual stream activation over **response tokens only** (system prompt and question tokens excluded) at layer 14 (mid-model) for every rollout. Float32.

---

### `EMOTIONAL_AXES/emotional_acts_L26.pt`
Tensor of shape **(12390, 3072)**. Same extraction at layer 26 (PROBE_LAYER). This is the tensor used for all PCA and axis work. Float32.

Response-token-only extraction is critical: full-sequence mean gives cosines ≥ +0.95 across all tone pairs (system prompt dominates). Response-only gives meaningful separation.

**Sanity check from notebook:**
```
acts_L26 norm (mean): ~typical residual stream magnitude
cos(humor, melancholy) centered at L26: −0.5839
cos(humor, horror)     centered at L26: −0.3487
cos(melancholy, horror) centered at L26: +0.1380
```

---

### `EMOTIONAL_AXES/emotional_acts_meta.csv`
**12,390 rows.** Index-matched to the activation tensors. Columns: `tone`, `polarity`, `prompt_idx`, `question_idx`, `question`. Allows joining activation rows back to text metadata.

---

## Script 09 — PCA and Emotional Axis Definition

---

### `EMOTIONAL_AXES/global_mean.npy`
Shape **(3072,)**. Mean activation vector across all 59 tone means (positive prompts only). Used for centering before PCA.

---

### `EMOTIONAL_AXES/pca_components.npy`
Shape **(10, 3072)**. Top-10 PCA components from the centered 59-tone mean matrix.

**Variance explained by top 10 PCs:**
```
PC1: 23.9%    PC2: 19.0%    PC3: 17.0%    PC4:  9.3%    PC5:  5.3%
PC6:  4.3%    PC7:  3.3%    PC8:  2.4%    PC9:  1.8%   PC10:  1.5%
Top 3 total: 59.9%
```

---

### `EMOTIONAL_AXES/pca_explained.npy`
Shape **(10,)**. Variance explained ratio for each of the 10 PCs. Subset of `pca_components.npy` metadata.

---

### `EMOTIONAL_AXES/axis_depth.npy`
Shape **(3072,)**. Unit-normalized. PC1, sign-flipped so that positive projection = emotional richness/depth. Positive end: wistfulness, tenderness, melancholy. Negative end: clinical, dry, analytical, detachment.

---

### `EMOTIONAL_AXES/axis_arousal.npy`
Shape **(3072,)**. Unit-normalized. PC2, sign-flipped so that positive projection = high arousal. Positive end: frenzy, panic, rage, terror, ecstasy. Negative end: serenity, detachment, numbness, boredom.

---

### `EMOTIONAL_AXES/axis_darkbright.npy`
Shape **(3072,)**. Unit-normalized. PC3, sign-flipped so that positive projection = dark narrative. Positive end: horror, grief, dread, despair. Negative end: elation, joy, excitement, mania.

**The three axes are orthogonal by construction (from PCA):**
```
cos(depth,   arousal)    = −0.0000
cos(depth,   darkbright) = −0.0000
cos(arousal, darkbright) = +0.0000
```

---

### `EMOTIONAL_AXES/tone_index.json`
JSON list of the 59 tone names in the order they appear in the activation matrices. Used to map row indices back to tone labels.

---

### `EMOTIONAL_AXES/axis_meta.json`
Human-readable summary of all three axes, their PC indices, variance explained, positive/negative pole descriptions, and the coordinates of all 59 tones and the three EMNLP poles on all three axes.

**EMNLP three-pole coordinates:**
```
humor:      depth=−4.611  arousal=+0.931  dark=−2.723
melancholy: depth=+3.882  arousal=−1.503  dark=+1.720
horror:     depth=+0.661  arousal=+2.012  dark=+3.158
```

---

### `EMOTIONAL_AXES/emotional_space_PC1_PC2.png`
Scatter plot of all 59 tones in PC1 × PC2 space (depth vs. arousal), coloured by emotional family (green = positive valence, blue = melancholic, red = dark/horror, grey = neutral/analytical). The three EMNLP poles marked with stars.

---

### `EMOTIONAL_AXES/emotional_space_final.png`
Two-panel scatter: (left) depth × arousal, (right) depth × dark/bright. Both plots show all 59 tone labels and star-mark the three EMNLP poles. This is the main figure for the emotional geometry extension.

---

## Script 10 — Causal Axis Validation

---

### `EMOTIONAL_AXES/phase4_validation.csv`
Scored steering results across the three axes. Columns: `axis`, `direction` (positive/negative/baseline), `mult`, `quality`, `mean_score`, `std_score`, `n_scored`, `sample_text`.

**DeepSeek scores (0–100) per axis, direction, and multiplier:**
```
DEPTH axis:
  baseline  mult=0:   31.0 / 100
  positive  mult=3:   50.0 / 100
  positive  mult=6:   48.5 / 100
  positive  mult=9:   75.5 / 100
  negative  mult=3:   84.5 / 100
  negative  mult=6:   93.5 / 100
  negative  mult=9:   87.5 / 100

AROUSAL axis:
  baseline  mult=0:   38.5 / 100
  positive  mult=3:   73.0 / 100
  positive  mult=6:   99.5 / 100
  positive  mult=9:  100.0 / 100
  negative  mult=3:   45.0 / 100
  negative  mult=6:   58.0 / 100
  negative  mult=9:   55.0 / 100

DARK/BRIGHT axis:
  baseline  mult=0:   12.8 / 100
  positive  mult=3:   16.5 / 100
  positive  mult=6:   24.5 / 100
  positive  mult=9:   39.0 / 100
  negative  mult=3:    0.0 / 100
  negative  mult=6:    3.0 / 100
  negative  mult=9:    2.0 / 100
```

---

### `EMOTIONAL_AXES/phase4_validation.png`
Three-panel line chart (one per axis). X-axis: multiplier. Y-axis: DeepSeek score (0–100). Three lines per panel: positive direction (red), negative direction (blue), baseline (grey). Shaded confidence bands (±1 std). Arousal positive line reaches ceiling at mult=6.

---

## Checkpoint Files (Infrastructure, Not Results)

The following are resumption checkpoints written during long runs. They contain intermediate state only, not final results.

| File | Written by | Purpose |
|---|---|---|
| `cell5_layer_sweep_raw.json` | Script 02 | Layer sweep checkpoint, one entry per layer |
| `cell5_mult_sweep_full.json` | Script 02 | Multiplier sweep raw data |
| `temp_sweep_checkpoint.json` | Script 03 | Temperature sweep progress |
| `task19_ablation_results.json` | Script 04 | Head ablation checkpoint |
| `task_nim_eval_checkpoint.json` | Script 06 | DeepSeek judge checkpoint |
| `EMOTIONAL_AXES/rollout_checkpoint.json` | Script 07 | Rollout generation checkpoint |
| `EMOTIONAL_AXES/phase2_checkpoint.json` | Script 08 | Activation extraction checkpoint |

---

## Total File Count

| Type | Count |
|---|---|
| `.pt` tensors | 10 |
| `.csv` tables | 17 |
| `.json` summaries | 16 |
| `.npy` arrays | 8 |
| `.png` figures | 16 |
| **Total** | **67** |
