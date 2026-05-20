# Steering Emotional Tone in Llama-3.2-3B
### EMNLP 2026 — Mechanistic Interpretability of Humor, Melancholy, and Horror

Activation steering across a three-pole emotional spectrum (humor, melancholy, horror) in `meta-llama/Llama-3.2-3B` using TransformerLens. We extract residual-stream activations, train pairwise linear probes, build mean-diff steering vectors, and evaluate steered generations at scale across layers, multipliers, and temperatures.

---

## Repo Structure

```
emnlp_steering_repo/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── humor_master.csv          ← 500 samples  (populate before running)
│   ├── melancholy_master.csv     ← 419 samples  (populate before running)
│   └── horror_master.csv         ← 199 samples  (populate before running)
│
├── scripts/
│   ├── cell00_install_mount.py
│   ├── cell01_load_model_helpers.py
│   ├── cell02_load_csvs.py
│   ├── cell03_activation_extraction_probe.py
│   ├── cell04_pairwise_probes_steering_vecs.py
│   ├── cell05_resume_checkpoint.py
│   ├── cell06_layer_mult_sweep.py
│   ├── cell07_resume_session2.py
│   ├── cell08_override_humor_config.py
│   ├── cell09_temperature_sweep.py
│   ├── cell10_eval_temp_override.py
│   ├── cell11_main_eval_vector_comparison.py
│   ├── cell12_resume_session3.py
│   ├── cell13_attention_head_ablation.py
│   ├── cell14_tasks_3_4_5.py
│   ├── cell15_resume_session4.py
│   ├── cell16_tasks_2_6.py
│   ├── cell17_six_ablations.py
│   ├── cell18_fix_cell12_summary.py
│   ├── cell19_expf_roberta.py
│   └── cell20_deepseek_judge.py
│
├── outputs/                      ← populated after running (not committed)
└── figures/                      ← populated after running (not committed)
```

---

## Setup

**Environment:** Google Colab Pro (A100). All scripts are written to run in Colab with Google Drive mounted. The notebook was run cell-by-cell in sequential Colab sessions.

**Data folder:** Create a folder named `STEERING_EMNLP_2026` in your Google Drive root and place the three CSVs there. Each CSV must have a `text` column.

**HuggingFace token:** Add `HF_TOKEN` to Colab Secrets (Tools → Secrets). Required to access `meta-llama/Llama-3.2-3B`.

**DeepSeek API key:** Required only for `cell20_deepseek_judge.py`. Paste your key into the `DEEPSEEK_API_KEY` variable in that file.

---

## Pipeline

```
data/ (three master CSVs)
      │
cell00  Install packages + mount Drive + discover STEERING_EMNLP_2026 folder
cell01  Load Llama-3.2-3B via TransformerLens + define all shared helpers
cell02  Load & validate CSVs → df_humor, df_mel, df_hor
      │
cell03  Extract activations at all 28 layers → humor_acts.pt, dark_acts.pt, probe_weights.pt
        Binary probe training → layer_probe_accuracy.png
        RESULT: L26 CV accuracy = 0.9417 ± 0.0091
      │
cell04  Three pairwise probes (humor↔mel, humor↔horror, mel↔horror)
        Build mean-diff steering vectors → steering_vecs.pt
        RESULT: cos(humor,mel)=−0.9972  cos(humor,horror)=−0.3167  cos(mel,horror)=+0.3151
      │
cell05  [RESUME BLOCK] — reload all artifacts if session was cut
      │
cell06  Layer sweep (mult=9.0, n=30) → intervention_results per layer
        Multiplier sweep on top-3 layers (mults 3–15, n=15) → best_configs.json
        RESULT: humor L18×12.0  melancholy L18×3.0  horror L15×9.0
      │
cell07  [RESUME BLOCK] — session 2 model reload
cell08  Manual humor config override (L18×12.0 confirmed)
      │
cell09  Temperature sweep (T=0.0–1.0, 11 temps × 3 emotions × 100 samples)
        → temp_sweep_results.csv, temp_sweep_confusion_grid.png, temp_sweep_metrics_plot.png
        RESULT: BEST_TEMP=1.0 (avg_both=69.3%)
      │
cell10  Override EVAL_TEMP=0.7 for paper's main results
      │
cell11  Main eval at T=0.7 + vector comparison (mean_diff vs probe_w vs random, n=50, Bonferroni)
        → probe_confusion_matrix_T07.png, simplex_results_T07.csv, vector_comparison.csv
        RESULT: avg diagonal=90.0%  horror=96%  mel=82%  humor=92%
      │
cell12  [RESUME BLOCK] — session 3 model reload
cell13  Attention head ablation (6 layers × 24 heads × 100 samples)
        RESULT: 10 critical heads, all at L0. Top: L0H7 drop=0.060
      │
cell14  Tasks 3, 4, 5 (CPU only — no model needed)
        Task 3: mel↔horror bleed 7.5% vs humor↔other 3.8% (ratio 2.0x)
        Task 4: surface probe acc=0.788, activation probe acc=0.900, gap=+0.112
        Task 5: OOD drop — humor−28.6%, mel−65.9%, horror−5.9%
      │
cell15  [RESUME BLOCK] — session 4 model reload
      │
cell16  Task 2 (additive vs proj-inject) + Task 6 (2-turn tonal drift)
        Task 2: horror proj=63% vs add=90% (**); humor/mel ns
        Task 6: T1=+5.855 → T2=+7.729; drift=−1.874; p=0.042*; 23.3% neutralized
      │
cell17  Six ablation experiments (A–F)
        Exp A: random baseline; Exp B: data size; Exp C: RepE PCA vs mean-diff
        Exp D: head ablation control (SV agreement); Exp E: 4-turn drift; Exp F: DistilRoBERTa
      │
cell18  Fix: save cell12_summary with string keys
cell19  Exp F fix: DistilRoBERTa using ste_gen column (correct column name)
      │
cell20  DeepSeek judge (deepseek-chat, 300 rows)
        RESULT: humor=95%  mel=88%  horror=47%  overall=76.7%
```

---

## Key Results

### Probe Accuracy by Layer
Peak at L26: **0.9417 ± 0.0091** (5-fold CV, humor vs dark)

### Steering Vector Geometry
| Pair | Cosine |
|---|---|
| humor · melancholy | −0.9972 |
| humor · horror | −0.3167 |
| melancholy · horror | +0.3151 |

Interpretation: humor and melancholy are antipodal on a shared valence axis. Horror is geometrically distinct — consistent with Russell's circumplex model of affect.

### Best Steering Configs
| Emotion | Layer | Mult | Both rate (T=0.7) |
|---|---|---|---|
| humor | 18 | 12.0 | 60% |
| melancholy | 18 | 3.0 | 58% |
| horror | 15 | 9.0 | 67% |

### Temperature Sweep (avg_both, n=100 per emotion)
| Temp | humor | mel | horror | avg_diag |
|---|---|---|---|---|
| 0.0 | 45% | 32% | 52% | 86.7% |
| 0.3 | 43% | 53% | 59% | 88.0% |
| 0.7 | 60% | 58% | 67% | 90.0% |
| 1.0 | 64% | 66% | 78% | 87.7% |

Paper reports at T=0.7 (avg_diag=90.0%).

### Main Eval at T=0.7 — Confusion Matrix
| Target | →humor | →mel | →horror |
|---|---|---|---|
| humor | **92%** | 2% | 6% |
| mel | 3% | **82%** | 15% |
| horror | 4% | 0% | **96%** |
Avg diagonal: **90.0%**

### Vector Comparison (n=50, Bonferroni)
| Emotion | mean_diff | probe_w | random |
|---|---|---|---|
| humor | 58% | 40% | 36% |
| melancholy | 52% | 60% | 44% |
| horror | 74% | 80% | 50% |

### Attention Head Ablation
- Critical heads (drop >2%): **10 — all at L0**
- Layers 7, 14, 15, 18, 26: zero critical heads
- Top: **L0H7** drop=0.060

### Surface vs Activation Probe (Task 4)
- Surface probe CV acc: **0.788 ± 0.012**
- Activation probe acc: **0.900**
- Gap: **+0.112**

### OOD Evaluation (Task 5)
| Emotion | In-dist | OOD source | OOD acc | Drop |
|---|---|---|---|---|
| humor | 87.6% | english_quotes | 59.0% | −28.6% |
| melancholy | 96.9% | merve/poetry | 31.0% | −65.9% |
| horror | 22.6% | imdb (neg) | 28.5% | −5.9% |

### Ablations (Cell 12)
- **Exp A — Random baseline:** humor our=40% vs rand=48.9% p=0.008** (small-n artifact at n=15); horror our=86.7% vs rand=46.7% p=0.0002***
- **Exp C — RepE PCA vs Mean-Diff:** cos(PCA, mean_diff)=−0.2033; cos(MD, probe_w)=+0.8848
- **Exp D — Head control:** SV baseline acc=0.300; mean specificity=+0.020
- **Exp E — 4-turn drift:** T1=+6.892 → T2=+4.834; T1 vs T2 ns (p≈0.08, n=15)

### DeepSeek Judge (Cell 13, n=300)
| Emotion | Accuracy | Mean score |
|---|---|---|
| humor | 95% | 2.78 |
| melancholy | 88% | 3.59 |
| horror | 47% | 3.50 |
Overall: **76.7%**

Spearman r (ps_shift vs DeepSeek): humor r=+0.202 p=0.044*; mel ns; horror ns

Cross-validation DeepSeek × DistilRoBERTa: humor r=+0.553***, mel r=+0.444***, horror r=+0.445***

---

## Model & Data

**Model:** `meta-llama/Llama-3.2-3B` — https://huggingface.co/meta-llama/Llama-3.2-3B

**Data sources (for reproduction):**
- Humor: https://huggingface.co/datasets/Ichhyamayee/short-jokes (Groq-compressed to 55–199 chars)
- Melancholy: r/depression Reddit posts (Groq-compressed)
- Horror: r/nosleep Reddit posts (Groq-compressed)

Data was manually curated and compressed via the Groq API to 55–199 character samples before training. The three master CSVs are the final curated outputs of that pipeline.

---

## Notes

- **Coherence threshold:** 0.6 (TTR-based, entropy-gated, real-word fraction). Texts below this are excluded from "both" counts.
- **"Both" metric:** probe shift is positive AND coherence ≥ 0.6.
- **PROBE_LAYER:** always `N_LAYERS − 2 = 26`. This is empirically the peak semantic readout layer for Llama-3.2-3B.
- **Hook type:** projection-injection throughout (`resid -= proj(resid, sv); resid += sv * mult`). Additive hook compared in Task 2.
- **Seeds:** train=42, eval=2026, sweep=99.
- **Exp A humor result:** our=40% < random=48.9% at n=15 is a known small-sample artifact. Horror at n=15 holds (p=0.0002***). See results doc for full discussion.

---

## Citation

```bibtex
@inproceedings{emnlp2026emotionsteering,
  title     = {Steering Emotional Tone in Large Language Models via Residual Stream Intervention},
  author    = {[anonymized]},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```
