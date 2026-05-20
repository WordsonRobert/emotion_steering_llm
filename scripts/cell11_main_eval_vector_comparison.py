# cell11_main_eval_vector_comparison.py
#
# Part A: Confusion matrix + per-emotion summary at T=0.7.
#         This is the paper's primary result table.
#
# Part B: Vector comparison — mean_diff vs probe_w vs random (null baseline).
#         n=50 per vector per emotion. Pairwise Mann-Whitney U with Bonferroni correction.
#
# RESULT (Part A):
#   avg diagonal = 90.0%
#   humor=92%, mel=82%, horror=96%
#
# RESULT (Part B, n=50, Bonferroni):
#   humor:  mean_diff=58%  probe_w=40%  random=36%  (mean_diff vs random ***)
#   mel:    mean_diff=52%  probe_w=60%  random=44%  (ns)
#   horror: mean_diff=74%  probe_w=80%  random=50%  (**)
#
# Requires (from cell09/cell10):
#   model, device, EMOTIONS, PROBE_LAYER, DRIVE
#   df_sweep, df_results (T=0.7)
#   sv, w_hm, w_hh, PROBE_FOR, PROBE_SIGN, CONFIGS
#   coherence_score, generate, build_hook
#
# Runtime: ~1.5 hrs on A100

import torch, numpy as np, pandas as pd, json, os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from itertools import combinations

EVAL_TEMP = 0.7
DRIVE     = '/content/drive/MyDrive/STEERING_EMNLP_2026'

df_results = df_sweep[df_sweep['temperature'] == EVAL_TEMP].copy()
df_results.to_csv(f'{DRIVE}/simplex_results_T07.csv', index=False)
print(f'Working at T={EVAL_TEMP}  ({len(df_results)} rows)')


# ── Part A: Confusion matrix ──────────────────────────────────────

conf = np.zeros((3, 3))
for i, true_em in enumerate(EMOTIONS):
    sub = df_results[df_results['emotion_target'] == true_em]
    for j, pred_em in enumerate(EMOTIONS):
        conf[i, j] = (sub['pred_emotion'] == pred_em).sum() / max(len(sub), 1)

print(f'\n{"="*60}')
print(f'PROBE CONFUSION MATRIX  (T={EVAL_TEMP})')
print(f'{"="*60}')
print(f'  {"":12} {"→humor":>10} {"→mel":>10} {"→horror":>10}')
for i, em in enumerate(EMOTIONS):
    print(f'  {em:12} {conf[i,0]:>10.1%} {conf[i,1]:>10.1%} {conf[i,2]:>10.1%}')
print(f'  Avg diagonal: {np.diag(conf).mean():.1%}')

for emotion in EMOTIONS:
    sub = df_results[df_results['emotion_target'] == emotion]
    print(f'\n  {emotion.upper()} — L{CONFIGS[emotion]["layer"]} ×{CONFIGS[emotion]["mult"]}:')
    print(f'    Both:       {sub["both"].mean():.1%}')
    print(f'    Pred corr:  {sub["pred_correct"].mean():.1%}')
    print(f'    Coherent:   {sub["is_coherent"].mean():.1%}')
    print(f'    Mean shift: {sub["ps_shift"].mean():+.4f} ± {sub["ps_shift"].std():.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(conf, annot=True, fmt='.1%',
            xticklabels=['→humor','→mel','→horror'], yticklabels=['humor','melancholy','horror'],
            cmap='Blues', ax=axes[0], vmin=0, vmax=1, linewidths=0.5)
axes[0].set_title(f'Probe Confusion Matrix\nT={EVAL_TEMP}  avg_diag={np.diag(conf).mean():.1%}', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Steering target')

score_matrix = np.array([
    [df_results[df_results['emotion_target']==em]['score_humor'].mean(),
     df_results[df_results['emotion_target']==em]['score_mel'].mean(),
     df_results[df_results['emotion_target']==em]['score_horror'].mean()]
    for em in EMOTIONS
])
sns.heatmap(score_matrix, annot=True, fmt='.3f',
            xticklabels=['humor score','mel score','horror score'],
            yticklabels=['→humor','→mel','→horror'],
            cmap='RdYlGn', ax=axes[1], linewidths=0.5, center=0)
axes[1].set_title('Mean Probe Scores per Steering Target', fontsize=11, fontweight='bold')
fig.suptitle(f'Three-Pole Steering — Llama-3.2-3B  |  T={EVAL_TEMP}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE}/probe_confusion_matrix_T07.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'✅ probe_confusion_matrix_T07.png saved')


# ── Part B: Vector comparison ─────────────────────────────────────

print(f'\n{"="*60}')
print('PART B: VECTOR COMPARISON')
print('  Comparing: mean_diff | probe_w | random  (n=50, T=0.7, Bonferroni)')

VEC_N    = 50
actual_t = 1e-10 if EVAL_TEMP == 0.0 else EVAL_TEMP

def _build_hook_from_vec(vec, emotion):
    cfg = CONFIGS[emotion]; mult = float(cfg['mult'])
    def fn(resid_pre, hook):
        v = vec; proj = (resid_pre @ v).unsqueeze(-1) * v
        return (resid_pre - proj) + v * mult
    return [(f'blocks.{cfg["layer"]}.hook_resid_pre', fn)]

def _score_gen(ste_out, w, sign):
    tokens = model.to_tokens([ste_out], prepend_bos=True)
    if tokens.shape[1] > 512: tokens = tokens[:, -512:]
    hn = f'blocks.{PROBE_LAYER}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: hn in n, stop_at_layer=PROBE_LAYER+1)
    act = cache[hn][0, -1, :].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return sign * float(act @ w)

def _run_vec_eval(emotion, vec, vec_name, prompts):
    w = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
    hooks = _build_hook_from_vec(vec, emotion)
    results = []
    for prompt in prompts:
        p = prompt[:120]
        base_out = model.generate(p, max_new_tokens=60, temperature=actual_t, verbose=False)
        base_ps  = _score_gen(base_out, w, sign)
        with model.hooks(fwd_hooks=hooks):
            ste_out = model.generate(p, max_new_tokens=60, temperature=actual_t, verbose=False)
        ste_ps  = _score_gen(ste_out, w, sign)
        ste_gen = ste_out[len(p):].strip()
        coh     = coherence_score(ste_gen)
        results.append({'both': bool((ste_ps > base_ps) and (coh >= 0.6)),
                        'ps_shift': float(ste_ps - base_ps), 'coherence': float(coh),
                        'vec': vec_name, 'emotion': emotion})
    return results

vec_sets = {}
for emotion in EMOTIONS:
    md  = sv[emotion].cpu().float()
    if emotion == 'humor': pw_np = w_hm.copy()
    elif emotion == 'melancholy': pw_np = -w_hm.copy()
    else: pw_np = -w_hh.copy()
    pw  = torch.tensor(pw_np, dtype=torch.float32); pw = pw / pw.norm().clamp(min=1e-8)
    rnd = torch.randn_like(md); rnd = rnd / rnd.norm().clamp(min=1e-8)
    vec_sets[emotion] = {
        'mean_diff': md.to(device).to(torch.bfloat16),
        'probe_w':   pw.to(device).to(torch.bfloat16),
        'random':    rnd.to(device).to(torch.bfloat16),
    }

eval_prompts_vc = {
    'humor':      df_humor.sample(VEC_N, random_state=77).reset_index(drop=True)['text'].tolist(),
    'melancholy': df_mel.sample(VEC_N,   random_state=77).reset_index(drop=True)['text'].tolist(),
    'horror':     df_hor.sample(VEC_N,   random_state=77).reset_index(drop=True)['text'].tolist(),
}

vc_rows = []
for emotion in EMOTIONS:
    prompts = eval_prompts_vc[emotion]
    for vec_name, vec in vec_sets[emotion].items():
        print(f'  {emotion:12} {vec_name:12} ...', end='', flush=True)
        rows = _run_vec_eval(emotion, vec, vec_name, prompts)
        vc_rows.extend(rows)
        print(f'  both={np.mean([r["both"] for r in rows]):.1%}')

df_vc = pd.DataFrame(vc_rows)
df_vc.to_csv(f'{DRIVE}/vector_comparison.csv', index=False)

print(f'\n{"="*60}')
print('VECTOR COMPARISON SUMMARY')
print(f'  {"emotion":12} {"mean_diff":>12} {"probe_w":>10} {"random":>10}')
for emotion in EMOTIONS:
    sub = df_vc[df_vc['emotion'] == emotion]
    vals = {vn: sub[sub['vec'] == vn]['both'].mean() for vn in ['mean_diff', 'probe_w', 'random']}
    print(f'  {emotion:12} {vals["mean_diff"]:>12.1%} {vals["probe_w"]:>10.1%} {vals["random"]:>10.1%}')

vec_names      = ['mean_diff', 'probe_w', 'random']
all_pairs      = list(combinations(vec_names, 2))
n_comparisons  = len(EMOTIONS) * len(all_pairs)

print(f'\n{"="*60}')
print('PAIRWISE MANN-WHITNEY U + BONFERRONI CORRECTION')
pval_results = []
for emotion in EMOTIONS:
    sub = df_vc[df_vc['emotion'] == emotion]
    for v1, v2 in all_pairs:
        s1   = sub[sub['vec'] == v1]['ps_shift'].values
        s2   = sub[sub['vec'] == v2]['ps_shift'].values
        stat, p = scipy_stats.mannwhitneyu(s1, s2, alternative='two-sided')
        p_adj = min(p * n_comparisons, 1.0)
        sig   = ('***' if p_adj < 0.001 else ('**' if p_adj < 0.01 else ('*' if p_adj < 0.05 else 'ns')))
        pval_results.append({'emotion': emotion, 'v1': v1, 'v2': v2,
                             'U': float(stat), 'p_raw': float(p), 'p_bonf': float(p_adj), 'sig': sig})
        print(f'  {emotion:12} {v1:12} vs {v2:12}  U={stat:.0f}  p_bonf={p_adj:.4f}  {sig}')

pd.DataFrame(pval_results).to_csv(f'{DRIVE}/vector_comparison_stats.csv', index=False)

cell7_summary = {
    'eval_temp': EVAL_TEMP,
    'avg_diagonal_T07': float(np.diag(conf).mean()),
    'per_emotion_T07': {
        em: {
            'both':         float(df_results[df_results['emotion_target']==em]['both'].mean()),
            'pred_correct': float(df_results[df_results['emotion_target']==em]['pred_correct'].mean()),
            'coherent':     float(df_results[df_results['emotion_target']==em]['is_coherent'].mean()),
            'mean_shift':   float(df_results[df_results['emotion_target']==em]['ps_shift'].mean()),
        } for em in EMOTIONS
    },
    'vector_comparison': {
        em: {vn: float(df_vc[(df_vc['emotion']==em) & (df_vc['vec']==vn)]['both'].mean())
             for vn in vec_names} for em in EMOTIONS
    },
    'bonferroni_n_comparisons': n_comparisons,
}
with open(f'{DRIVE}/cell7_summary.json', 'w') as f:
    json.dump(cell7_summary, f, indent=2)
print('✅ cell7_summary.json saved')
print(f'\n✅ Cell 11 complete')
