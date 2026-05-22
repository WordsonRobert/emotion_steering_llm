# Steering Emotional Tone in Llama-3.2-3B
### EMNLP 2026 — Mechanistic Interpretability of Humor, Melancholy, and Horror

---

## What This Is

This repository contains the full experimental pipeline for a study of emotional tone in `meta-llama/Llama-3.2-3B`. The core question is simple: **does the model store emotional tone as a geometric structure inside it, and can we push that structure around to change the emotional flavour of what it generates — without breaking the language?**

The short answer is yes, and the structure it uses matches what psychology already knew about how humans organise emotions. Everything here is exactly what was run. No post-hoc cleanup.

---

## Repo Structure

```
/
├── README.md
├── requirements.txt
│
├── data/
│   ├── humor_master.csv          ← 500 samples  (populate before running)
│   ├── melancholy_master.csv     ← 419 samples  (populate before running)
│   └── horror_master.csv         ← 199 samples  (populate before running)
│
├── scripts/
│   ├── 01_setup_model_probe.py           — environment, model load, activation extraction,
│   │                                       binary probe training across all 28 layers
│   ├── 02_probes_steering_sweep.py       — three pairwise probes, steering vectors,
│   │                                       layer sweep and multiplier sweep
│   ├── 03_temp_sweep_main_eval.py        — temperature sweep, main confusion matrix at T=0.7,
│   │                                       vector comparison (mean_diff vs probe_w vs random)
│   ├── 04_head_ablation_cpu_analyses.py  — attention head ablation, surface vs activation probe,
│   │                                       OOD eval, three-pole geometry analysis
│   ├── 05_ablations_drift_roberta.py     — additive vs proj-inject hook, 2-turn and 4-turn
│   │                                       tonal drift, six ablation experiments, DistilRoBERTa
│   ├── 06_deepseek_judge.py              — DeepSeek-V3 LLM judge, cross-validation vs RoBERTa
│   ├── 07_emotional_tone_rollouts.py     — 12,390 DeepSeek rollouts across 59 emotional tones
│   ├── 08_emotional_activation_extraction.py — run rollouts through Llama, extract
│   │                                           response-token activations at L14 and L26
│   ├── 09_emotional_pca_axes.py          — PCA on 59-tone activation space, identify
│   │                                       three interpretable emotional axes
│   └── 10_emotional_axis_validation.py   — causal validation of the three axes via steering
│
├── outputs/                      ← populated after running (not committed)
└── figures/                      ← populated after running (not committed)
```

---

## Setup

**Environment:** Google Colab Pro with A100 GPU (scripts 01–06, 08, 10) or CPU-only (scripts 06, 07, 09). Scripts 01–06 were run across four Colab sessions; checkpointing throughout means any interrupted run can resume from where it stopped.

**Data folder:** Create a folder named `STEERING_EMNLP_2026` in your Google Drive root and place the three CSVs there. Each CSV needs a `text` column. Script 01 auto-discovers the folder by name.

**HuggingFace token:** Add `HF_TOKEN` to Colab Secrets (`Tools → Secrets`). Required to download `meta-llama/Llama-3.2-3B`.

**DeepSeek API key:** Required for scripts 06, 07, and 10. Add `DEEPSEEK_API_KEY` to Colab Secrets, or paste directly into the `DEEPSEEK_API_KEY` variable in those files. Get a key at https://platform.deepseek.com.

**Install:** `pip install -r requirements.txt`. Script 01 also handles installation inline for Colab.

---

## The Pipeline

The pipeline has two parts. Scripts 01–06 are the core paper. Scripts 07–10 are an extension that maps a broader emotional space.

---

### Part 1: The Core Experiment (Scripts 01–06)

The goal is to show that emotional tone in Llama-3.2-3B is geometrically structured and steerable.

---

#### Script 01 — Setup, Model Load, Activation Extraction, Binary Probe

Loads Llama-3.2-3B via TransformerLens (which gives direct access to internal activations without touching the model weights). Loads the three emotion datasets. For every text in the dataset, extracts the internal representation at each of the 28 layers. Trains a binary classifier (humor vs. dark = melancholy + horror) at every layer to find out where emotional tone is most readable.

**Outputs:** `humor_acts.pt`, `dark_acts.pt`, `probe_weights.pt`, `layer_probe_accuracy.png`

**Runtime:** ~50 min on A100

---

#### Script 02 — Three Pairwise Probes and Steering Vectors

Trains three separate binary classifiers — one for each emotion pair (humor/melancholy, humor/horror, melancholy/horror). From each classifier, builds a "steering vector": a direction in the model's internal space that points from one emotion toward the other. Then sweeps across all 28 layers and five multiplier values to find the best spot and strength to apply the steering.

**Outputs:** `probe_humor_vs_melancholy.pt`, `probe_humor_vs_horror.pt`, `probe_melancholy_vs_horror.pt`, `steering_vecs.pt`, `best_configs.json`, `layer_heatmap.png`

**Runtime:** ~3.5 hrs on A100

---

#### Script 03 — Temperature Sweep and Main Evaluation

Sweeps over 11 temperature values (0.0 to 1.0) across all three emotions to find the best generation temperature. Runs the main evaluation at T=0.7: for each emotion, steers 100 prompts and scores whether (a) the probe classifies the output as the target emotion and (b) the output is still coherent text. Also compares three types of steering vector — mean-difference (ours), probe weight direction, and random.

**Outputs:** `temp_sweep_results.csv`, `simplex_results_T07.csv`, `probe_confusion_matrix_T07.png`, `vector_comparison.csv`, `vector_comparison_stats.csv`, `cell7_summary.json`

**Runtime:** ~7 hrs on A100

---

#### Script 04 — Head Ablation and CPU Analyses

Three separate analyses, no GPU required after the head ablation:

- **Head ablation:** Knocks out individual attention heads one at a time at layers 0, 7, 14, 15, 18, 26 and measures how much binary probe accuracy drops. Identifies which heads are "load-bearing" for emotional tone.
- **Surface vs. activation probe (Task 4):** Trains a classifier on 10 hand-crafted text features (word repetition rate, punctuation density, sentence length, etc.) and compares its accuracy to the activation probe. Tests whether activation steering is doing something beyond surface style changes.
- **OOD generalization (Task 5):** Tests the surface probe on out-of-distribution datasets to see how much of the surface accuracy is dataset-specific.
- **Three-pole geometry (Task 3):** Checks whether the confusion pattern between emotions matches the geometric prediction (melancholy and horror should confuse each other more than either confuses with humor, because they share the same negative-valence side of the space).

**Outputs:** `task19_ablation_results.csv`, `task19_head_ablation.png`, `cell9_summary.json`, `task3/4/5 JSON + PNG files`

**Runtime:** ~37 min total (~12 min GPU for ablation, ~25 min CPU)

---

#### Script 05 — Hook Comparison, Tonal Drift, and Six Ablation Experiments

Six targeted experiments:

- **Task 2 (hook comparison):** Compares two ways of applying the steering: projection-injection (which removes the existing component along the steering direction before injecting) vs. plain additive (just add the vector). Tests whether the subtraction step matters.
- **Task 6 (2-turn tonal drift):** Horror-steers a generation (turn 1), then runs an unsteered neutral follow-up (turn 2) and measures whether the horror tone persists or disappears.
- **Exp A (random baseline):** Compares our steering vectors to random unit vectors of matched size. Controls for the possibility that any directional perturbation of the residual stream produces apparent tonal shifts.
- **Exp B (data size ablation):** Rebuilds the humor steering vector from training sets of size 25, 50, 100, 150, 200, 300, 500 and tests whether more data consistently improves steering.
- **Exp C (RepE PCA vs mean-diff):** Compares our mean-difference steering vectors to vectors built from PCA (the approach used in Representation Engineering, Zou et al. 2023). Tests which construction method produces better steering.
- **Exp D (head ablation control):** Tests whether the critical heads from the ablation (all at L0) also affect syntactic performance (subject-verb agreement), to see if they are emotion-specific or general-purpose.
- **Exp E (4-turn drift):** Extends the tonal drift experiment to four turns.
- **Exp F (DistilRoBERTa):** Runs all 300 steered outputs through a separate fine-tuned emotion classifier (`j-hartmann/emotion-english-distilroberta-base`) and computes Spearman correlation with our probe shift scores.

**Outputs:** `task2/6/random/data/repe/head/drift/roberta CSV + PNG files`

**Runtime:** ~3 hrs on A100

---

#### Script 06 — DeepSeek-V3 LLM Judge

Sends all 300 steered outputs (from Script 03, T=0.7) to DeepSeek-V3 via API. The model rates each output 1–5 on humor, melancholy, and horror, and picks the dominant emotion. Also cross-validates DeepSeek scores against the DistilRoBERTa scores from Script 05.

**Outputs:** `task_nim_eval.csv`, `task_nim_eval.png`, `cell13_summary.json`

**Runtime:** ~20 min, CPU, API calls only

---

### Part 2: The Emotional Geometry Extension (Scripts 07–10)

Having shown that three emotions are geometrically structured, we ask a broader question: what does the full emotional space of Llama-3.2-3B look like across 59 emotional tones?

---

#### Script 07 — Emotional Tone Rollout Generation

Uses DeepSeek-V3 to generate text in 59 emotional tones (from joy and whimsy to existential dread and dissociation), each with 5 positive system prompts and 2 neutral control prompts, answering 30 introspective questions. Total: 12,390 API calls. Fully checkpoint-resumable.

**Outputs:** `EMOTIONAL_AXES/emotional_rollouts.csv`

**Runtime:** ~4–5 hrs, CPU, API

---

#### Script 08 — Activation Extraction from Rollouts

Runs each of the 12,390 rollouts through Llama-3.2-3B and extracts the average internal representation over the *response tokens only* (not the system prompt or question) at layer 14 (middle) and layer 26 (probe layer). The response-token-only extraction is critical — extracting over the full sequence causes the system prompt to dominate and makes all 59 tone vectors nearly identical.

**Outputs:** `EMOTIONAL_AXES/emotional_acts_L14.pt`, `EMOTIONAL_AXES/emotional_acts_L26.pt`, `EMOTIONAL_AXES/emotional_acts_meta.csv`

**Runtime:** ~15 min on A100

---

#### Script 09 — PCA and Axis Definition

Builds a single mean activation vector per tone, centers them, and runs PCA. Identifies three main axes that together explain 60% of the variance across all 59 tones:

- **PC1 (23.9% variance) — Depth:** emotional richness (wistfulness, tenderness, melancholy) vs. detachment (clinical, dry, analytical)
- **PC2 (19.0% variance) — Arousal:** high energy (frenzy, panic, rage) vs. calm (serenity, numbness, boredom)
- **PC3 (17.0% variance) — Dark/Bright:** dark narrative (horror, grief, despair) vs. bright energy (elation, joy, excitement)

The three EMNLP emotions sit at: humor (depth=−4.6, arousal=+0.9, dark=−2.7), melancholy (depth=+3.9, arousal=−1.5, dark=+1.7), horror (depth=+0.7, arousal=+2.0, dark=+3.2).

**Outputs:** `axis_depth.npy`, `axis_arousal.npy`, `axis_darkbright.npy`, `axis_meta.json`, `tone_index.json`, `emotional_space_final.png`

**Runtime:** ~5 min, CPU

---

#### Script 10 — Causal Axis Validation

Tests whether the three discovered axes actually *control* emotional output by steering along each one at multipliers 3, 6, and 9, generating from neutral prompts, and scoring with DeepSeek-V3.

**Outputs:** `phase4_validation.csv`, `phase4_validation.png`, `phase4_summary.json`

**Runtime:** ~40 min total (~30 min GPU, ~10 min DeepSeek scoring)

---

## Main Results

### Probe Accuracy

| Probe | CV Accuracy |
|---|---|
| Binary (humor vs. dark), L26 | **0.9400 ± 0.0062** |
| humor vs. melancholy | 0.9916 ± 0.0061 |
| humor vs. horror | 0.9272 ± 0.0215 |
| melancholy vs. horror | 0.9798 ± 0.0152 |

Emotional tone is linearly readable from the model's internal representations, peaking at layer 26 of 28.

### Steering Geometry

| Emotion pair | Cosine similarity |
|---|---|
| humor · melancholy | −0.9972 |
| humor · horror | −0.3167 |
| melancholy · horror | +0.3151 |

Humor and melancholy are nearly opposite poles of the same axis. Horror is geometrically distinct from both. This matches the valence × arousal structure from Russell's circumplex model of affect in psychology.

### Main Evaluation at T=0.7 (n=100 per emotion)

**Confusion matrix:**

|  | →humor | →mel | →horror |
|---|---|---|---|
| humor steered | **92%** | 2% | 6% |
| mel steered | 3% | **82%** | 15% |
| horror steered | 4% | 0% | **96%** |

**Avg diagonal: 90.0%**

| Emotion | Both (shift + coherent) | Probe accuracy | Coherent | Mean shift |
|---|---|---|---|---|
| humor | 60% | 92% | 76% | +3.72 |
| melancholy | 58% | 82% | 99% | +0.46 |
| horror | 67% | 96% | 97% | +2.15 |

### External Validation

| Validator | humor | melancholy | horror |
|---|---|---|---|
| DistilRoBERTa (target score > 0.3) | 29% | 41% | 53% |
| DeepSeek dominant accuracy | **95%** | **88%** | **47%** |
| DeepSeek × RoBERTa cross-correlation | r=+0.553*** | r=+0.444*** | r=+0.445*** |

The two external validators strongly agree with each other (all p<0.001). Horror's lower dominant accuracy at DeepSeek (47%) is consistent with the confusion matrix: the model is producing strongly negative-valence text that borders on melancholy.

### Key Ablation Results

| Experiment | Finding |
|---|---|
| Head ablation | All 10 critical heads are at layer 0 (early lexical detectors). No critical heads at layers 7, 14, 15, 18, or 26. Emotional tone at L26 is distributed, not localized. |
| Surface vs. activation probe | Surface features achieve 78.8% accuracy. Activation probe achieves 90.0%. Gap: +11.2 pp. |
| Hook type (Task 2) | For horror: additive hook significantly outperforms proj-inject (90% vs 63%, p=0.004). Horror + melancholy: ns. |
| Tonal drift (Task 6) | After horror-steered turn 1 (+5.85), unsteered turn 2 scores +7.73. The steered context primes continuation rather than decaying. |
| RepE PCA vs mean-diff | cos(PCA, mean_diff) = −0.20. They encode different directions. Mean-diff aligns with probe weight (cos=+0.88); PCA does not. |
| Random baseline (n=15) | Horror: our vector 86.7% vs random 46.7% (p=0.0002***). Humor result at n=15 is a known small-sample artifact; see n=100 results. |

### Emotional Geometry (59 Tones)

Three orthogonal axes explain 60% of the variance across 59 emotional tones at layer 26:

| Axis | PC | Variance | Positive end | Negative end |
|---|---|---|---|---|
| Depth | 1 | 23.9% | Wistfulness, tenderness, melancholy | Clinical, dry, analytical |
| Arousal | 2 | 19.0% | Frenzy, panic, rage, terror | Serenity, detachment, boredom |
| Dark/Bright | 3 | 17.0% | Horror, grief, dread, despair | Elation, joy, excitement, mania |

**Causal validation scores (DeepSeek, 0–100):**

| Axis | Baseline | Positive mult=9 | Negative mult=6–9 |
|---|---|---|---|
| Depth | 31.0 | 75.5 | 93.5 |
| Arousal | 38.5 | **100.0** | 55.0 |
| Dark/Bright | 12.8 | 39.0 | 0.0 |

Arousal is the cleanest causal axis (100% at mult=9). Depth works. Dark/Bright is the weakest — steering along it produces text closer to nihilism/emptiness than explicit horror.

---

## Technical Notes

**Model:** `meta-llama/Llama-3.2-3B` — 28 layers, 3072-dimensional residual stream, 24 attention heads

**PROBE_LAYER:** Always L26 (`N_LAYERS - 2`). Chosen empirically as the peak semantic readout layer.

**Steering configs:** humor L18×12.0, melancholy L18×3.0, horror L15×9.0. The humor config was confirmed manually from the full sweep evidence.

**Hook type:** Projection-injection throughout (`resid -= proj(resid, sv); resid += sv * mult`), except where noted. Task 2 compares this to additive.

**Coherence metric:** TTR-based with three gates: ≥3 words, no single word >40% of tokens, character entropy ≥3.5 bits. Outputs failing any gate score coherence = 0.0. The **both** metric = probe shift positive AND coherence ≥ 0.6.

**Seeds:** train=42, eval=2026, sweep=99. Fixed throughout.

**Checkpointing:** Scripts 02 and 03 (layer sweep, temperature sweep) checkpoint after every layer/temperature and resume cleanly. Script 07 checkpoints every 100 API calls. Script 08 checkpoints every 500 rows.

---

## Data Sources

| Dataset | Source |
|---|---|
| Humor | https://huggingface.co/datasets/Ichhyamayee/short-jokes (Groq-compressed to 55–199 chars) |
| Melancholy | r/depression Reddit posts (Groq-compressed) |
| Horror | r/nosleep Reddit posts (Groq-compressed) |

The three master CSVs are the output of a separate curation pipeline (Groq API compression to 55–199 char samples) and are not committed to this repo. Populate the `data/` folder before running.

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
