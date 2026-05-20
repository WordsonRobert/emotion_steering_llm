# cell09_temperature_sweep.py
#
# Full temperature sweep: T ∈ {0.0, 0.1, ..., 1.0} × 3 emotions × 100 samples.
# Measures both-rate, probe classification accuracy, coherence, and TTR at each temp.
# Saves confusion grids, metrics plots, and raw CSV.
# Auto-selects BEST_TEMP = argmax avg_both and writes it to best_configs.json.
#
# Requires (from cell07/cell08):
#   model, device, EMOTIONS, DRIVE, PROBE_LAYER
#   df_humor, df_mel, df_hor
#   CONFIGS (with corrected humor L18×12.0)
#
# Produces:
#   df_sweep — full results DataFrame (3300 rows)
#   BEST_TEMP — 1.0 (max avg_both=69.3%)
#   Saved: temp_sweep_results.csv, temp_sweep_summary.json
#          temp_sweep_confusion_grid.png, temp_sweep_metrics_plot.png
#          simplex_results_final.csv (at BEST_TEMP)
#          best_configs.json updated with best_temp
#
# RESULT: BEST_TEMP=1.0, paper reports at T=0.7 (avg_diag=90.0%)
#
# Runtime: ~2 hrs on A100

import torch, numpy as np, pandas as pd, json, math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from contextlib import nullcontext
from tqdm import tqdm
from scipy import stats as scipy_stats

TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SAMPLE_N     = 100
PROBE_LAYER  = 26

w_hm = torch.load(f'{DRIVE}/probe_humor_vs_melancholy.pt').float().numpy()
w_hh = torch.load(f'{DRIVE}/probe_humor_vs_horror.pt').float().numpy()
steering_vecs = torch.load(f'{DRIVE}/steering_vecs.pt')
sv = {k: v.to(device).to(torch.bfloat16) for k, v in steering_vecs.items()}
with open(f'{DRIVE}/best_configs.json') as f:
    CONFIGS = json.load(f)

PROBE_FOR  = {'humor': w_hm, 'melancholy': w_hm, 'horror': w_hh}
PROBE_SIGN = {'humor': +1,   'melancholy': -1,    'horror': -1}

eval_sets = {
    'humor':      df_humor.sample(SAMPLE_N, random_state=2026).reset_index(drop=True),
    'melancholy': df_mel.sample(SAMPLE_N,   random_state=2026).reset_index(drop=True),
    'horror':     df_hor.sample(SAMPLE_N,   random_state=2026).reset_index(drop=True),
}
print(f"✅ Eval sets  |  {len(TEMPERATURES)} temps × 3 emotions × {SAMPLE_N} = {len(TEMPERATURES)*3*SAMPLE_N} samples")
print(f"   CONFIGS: {CONFIGS}")


def coherence_score(text):
    words = text.strip().split()
    if len(words) < 3: return 0.0
    counts = Counter(words)
    if counts.most_common(1)[0][1] / len(words) > 0.4: return 0.0
    chars = [c for c in text.lower() if c.isalpha()]
    if len(chars) < 10: return 0.0
    cc = Counter(chars)
    entropy = -sum((c/len(chars)) * math.log2(c/len(chars)) for c in cc.values())
    if entropy < 3.5: return 0.0
    G = ['ss-', 'ses', 'sse', 'sss', 'thed', 'bsp', 'antml']
    def is_real(w):
        w = w.strip('.,!?-()').lower()
        return len(w) < 2 or (any(c in 'aeiou' for c in w) and not any(p in w for p in G))
    return sum(is_real(w) for w in words) / len(words)

def type_token_ratio(text):
    words = text.strip().lower().split()
    return len(set(words)) / len(words) if words else 0.0

def rep_rate(text):
    words = text.strip().split()
    if not words: return 0.0
    return Counter(words).most_common(1)[0][1] / len(words)

def get_activation(text):
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > 512: tokens = tokens[:, -512:]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens,
            names_filter=lambda n: f'blocks.{PROBE_LAYER}.hook_resid_post' in n)
    act = cache[f'blocks.{PROBE_LAYER}.hook_resid_post'][0, -1, :].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return act

def probe_score(text, w, sign):
    return sign * float(get_activation(text) @ w)

def build_hook(emotion):
    cfg = CONFIGS[emotion]; svec = sv[emotion]
    layer = cfg['layer']; mult = cfg['mult']
    def fn(resid_pre, hook):
        proj = (resid_pre @ svec).unsqueeze(-1) * svec
        return (resid_pre - proj) + svec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]


# ── Main sweep ────────────────────────────────────────────────────

all_rows    = []
sweep_stats = {}
pbar = tqdm(total=len(TEMPERATURES)*3*SAMPLE_N, desc="Overall")

for temp in TEMPERATURES:
    print(f"\n── temp={temp} ──")
    sweep_stats[temp] = {}
    actual_temp = 1e-10 if temp == 0.0 else temp

    for emotion in EMOTIONS:
        hooks = build_hook(emotion); w = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
        edf   = eval_sets[emotion]
        for _, row in edf.iterrows():
            prompt   = str(row['text'])[:120]
            base_out = model.generate(prompt, max_new_tokens=40, temperature=actual_temp, verbose=False)
            base_gen = base_out[len(prompt):].strip()
            with model.hooks(fwd_hooks=hooks):
                ste_out = model.generate(prompt, max_new_tokens=40, temperature=actual_temp, verbose=False)
            ste_gen  = ste_out[len(prompt):].strip()
            ste_coh  = coherence_score(ste_gen)
            is_coh   = ste_coh >= 0.6
            base_ps  = probe_score(base_out, w, sign)
            ste_ps   = probe_score(ste_out,  w, sign)
            ps_shift = ste_ps - base_ps
            act      = get_activation(ste_out)
            s_h      =  float(act @ w_hm)
            s_m      = -float(act @ w_hm)
            s_r      = -float(act @ w_hh)
            pred_em  = EMOTIONS[np.argmax([s_h, s_m, s_r])]
            scores_sorted = sorted([s_h, s_m, s_r], reverse=True)
            margin   = scores_sorted[0] - scores_sorted[1]
            all_rows.append({
                'temperature': temp, 'emotion_target': emotion, 'prompt': prompt[:60],
                'base_gen': base_gen, 'ste_gen': ste_gen,
                'ste_coh': ste_coh, 'is_coherent': is_coh,
                'ttr': type_token_ratio(ste_gen), 'rep_rate': rep_rate(ste_gen),
                'ps_shift': ps_shift, 'ps_success': ste_ps > base_ps,
                'both': (ste_ps > base_ps) and is_coh,
                'score_humor': s_h, 'score_mel': s_m, 'score_horror': s_r,
                'pred_emotion': pred_em, 'pred_correct': pred_em == emotion, 'pred_margin': margin,
            })
            pbar.update(1)

        em_rows = [r for r in all_rows if r['temperature']==temp and r['emotion_target']==emotion]
        sweep_stats[temp][emotion] = {
            'both':          float(np.mean([r['both']         for r in em_rows])),
            'pred_correct':  float(np.mean([r['pred_correct'] for r in em_rows])),
            'coherent':      float(np.mean([r['is_coherent']  for r in em_rows])),
            'mean_shift':    float(np.mean([r['ps_shift']     for r in em_rows])),
            'mean_ttr':      float(np.mean([r['ttr']          for r in em_rows])),
            'mean_rep_rate': float(np.mean([r['rep_rate']     for r in em_rows])),
            'mean_margin':   float(np.mean([r['pred_margin']  for r in em_rows])),
        }
        s = sweep_stats[temp][emotion]
        print(f"  {emotion:12s} both={s['both']:.1%}  pred={s['pred_correct']:.1%}  "
              f"coh={s['coherent']:.1%}  shift={s['mean_shift']:+.3f}")

pbar.close()
df_sweep = pd.DataFrame(all_rows)
df_sweep.to_csv(f'{DRIVE}/temp_sweep_results.csv', index=False)
with open(f'{DRIVE}/temp_sweep_summary.json', 'w') as f:
    json.dump(sweep_stats, f, indent=2)
print(f"\n✅ temp_sweep_results.csv saved ({len(df_sweep)} rows)")


# ── Confusion grid ────────────────────────────────────────────────

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes_flat = axes.flatten()
for idx, temp in enumerate(TEMPERATURES):
    ax  = axes_flat[idx]
    sub = df_sweep[df_sweep['temperature'] == temp]
    conf = np.zeros((3, 3))
    for i, te in enumerate(EMOTIONS):
        s2 = sub[sub['emotion_target'] == te]
        for j, pe in enumerate(EMOTIONS):
            conf[i, j] = (s2['pred_emotion'] == pe).sum() / len(s2)
    sns.heatmap(conf, annot=True, fmt='.0%', xticklabels=['H','M','R'],
                yticklabels=['hum','mel','hor'], cmap='Blues', ax=ax, vmin=0, vmax=1,
                linewidths=0.5, cbar=False, annot_kws={'size': 8})
    ax.set_title(f'temp={temp}  diag={np.diag(conf).mean():.0%}', fontsize=9)
for idx in range(len(TEMPERATURES), len(axes_flat)):
    axes_flat[idx].axis('off')
fig.suptitle('Confusion Matrices Across Temperatures — Llama-3.2-3B', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE}/temp_sweep_confusion_grid.png', dpi=130, bbox_inches='tight')
print("✅ temp_sweep_confusion_grid.png saved")
plt.close()


# ── Metrics plots ─────────────────────────────────────────────────

COLORS = {'humor': '#2196F3', 'melancholy': '#9C27B0', 'horror': '#F44336'}
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
METRICS = [
    ('both',         'Both (success + coherent)', axes[0,0]),
    ('pred_correct', 'Probe Classification Accuracy', axes[0,1]),
    ('coherent',     'Coherence Rate', axes[0,2]),
    ('mean_shift',   'Mean Probe Shift', axes[1,0]),
    ('mean_ttr',     'Type-Token Ratio', axes[1,1]),
    ('mean_rep_rate','Repetition Rate', axes[1,2]),
]
for key, label, ax in METRICS:
    for emotion in EMOTIONS:
        vals = [sweep_stats[t][emotion][key] for t in TEMPERATURES]
        ax.plot(TEMPERATURES, vals, '-o', color=COLORS[emotion], label=emotion, linewidth=2, markersize=5)
    ax.set_xlabel('Temperature'); ax.set_ylabel(label)
    ax.set_title(label, fontsize=10); ax.legend(fontsize=8)
    ax.set_xticks(TEMPERATURES); ax.grid(alpha=0.3)
    if key in ['both', 'pred_correct', 'coherent']: ax.set_ylim(0, 1.05)
fig.suptitle('Steering Metrics Across Temperatures — Llama-3.2-3B', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE}/temp_sweep_metrics_plot.png', dpi=130, bbox_inches='tight')
print("✅ temp_sweep_metrics_plot.png saved")
plt.close()


# ── Statistical tests ─────────────────────────────────────────────

print("\n" + "="*60)
print("STATISTICAL TESTS — low temp (≤0.3) vs high temp (≥0.7)")
low_temps  = [t for t in TEMPERATURES if t <= 0.3]
high_temps = [t for t in TEMPERATURES if t >= 0.7]
for emotion in EMOTIONS:
    sub_em  = df_sweep[df_sweep['emotion_target'] == emotion]
    low_df  = sub_em[sub_em['temperature'].isin(low_temps)]
    high_df = sub_em[sub_em['temperature'].isin(high_temps)]
    stat, p = scipy_stats.mannwhitneyu(low_df['ps_shift'].values, high_df['ps_shift'].values, alternative='two-sided')
    sig = 'significant' if p < 0.05 else 'not significant'
    print(f"  {emotion}: U={stat:.0f}  p={p:.4f}  ({sig})")


# ── Summary table + BEST_TEMP ─────────────────────────────────────

print("\n" + "="*70)
print("FINAL SUMMARY TABLE")
print(f"{'temp':>6} {'h_both':>8} {'m_both':>8} {'r_both':>8} {'avg_diag':>9}")
for temp in TEMPERATURES:
    h = sweep_stats[temp]['humor']; m = sweep_stats[temp]['melancholy']; r = sweep_stats[temp]['horror']
    avg = np.mean([h['pred_correct'], m['pred_correct'], r['pred_correct']])
    print(f"{temp:>6.1f} {h['both']:>8.1%} {m['both']:>8.1%} {r['both']:>8.1%} {avg:>9.1%}")

avg_both_per_temp = {temp: np.mean([sweep_stats[temp][em]['both'] for em in EMOTIONS]) for temp in TEMPERATURES}
BEST_TEMP = max(avg_both_per_temp, key=avg_both_per_temp.get)
print(f"\n✅ BEST_TEMP = {BEST_TEMP}  (avg_both={avg_both_per_temp[BEST_TEMP]:.1%})")

with open(f'{DRIVE}/best_configs.json') as f:
    cfg_file = json.load(f)
cfg_file['best_temp'] = BEST_TEMP
with open(f'{DRIVE}/best_configs.json', 'w') as f:
    json.dump(cfg_file, f, indent=2)

df_results = df_sweep[df_sweep['temperature'] == BEST_TEMP].copy()
df_results.to_csv(f'{DRIVE}/simplex_results_final.csv', index=False)
print(f"✅ simplex_results_final.csv saved ({len(df_results)} rows at BEST_TEMP={BEST_TEMP})")
print("🎉 TEMPERATURE SWEEP COMPLETE")
