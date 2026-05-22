#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 3: TEMPERATURE SWEEP + MAIN EVAL AT T=0.7
# ═══════════════════════════════════════════════════════════════════
# Covers notebook Cells 6-7 (temp sweep + main eval + vector comparison)
#
# Requires Scripts 1-2 to have completed
#
# What this does:
#   Cell 6: Temperature sweep 0.0→1.0 across all three emotions
#            → Finds BEST_TEMP, saves simplex_results_final.csv
#   Cell 7: Main confusion matrix at T=0.7 (paper's primary result)
#            + Vector comparison (mean_diff vs probe_w vs random)
#              with Bonferroni correction
#
# Runtime:  ~7 hrs on A100 (Cell 6: ~6 hrs, Cell 7: ~1 hr)
# Outputs:  temp_sweep_results.csv, simplex_results_T07.csv,
#           probe_confusion_matrix_T07.png, vector_comparison.csv,
#           vector_comparison_stats.csv, cell7_summary.json
# ═══════════════════════════════════════════════════════════════════

import os, gc, json, math, torch
import numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from scipy import stats as scipy_stats
from itertools import combinations
from transformer_lens import HookedTransformer
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

DRIVE    = '/content/drive/MyDrive/STEERING_EMNLP_2026'
device   = 'cuda'
EMOTIONS = ['humor', 'melancholy', 'horror']

model = HookedTransformer.from_pretrained(
    'meta-llama/Llama-3.2-3B', dtype=torch.bfloat16, device=device)
model.eval()

N_LAYERS    = model.cfg.n_layers
D_MODEL     = model.cfg.d_model
PROBE_LAYER = N_LAYERS - 2

df_humor = pd.read_csv(f'{DRIVE}/humor_master.csv')
df_mel   = pd.read_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor, df_mel, df_hor]:
    df['text'] = df['text'].fillna('').astype(str).str.strip()

w_hm = torch.load(f'{DRIVE}/probe_humor_vs_melancholy.pt').float().numpy()
w_hh = torch.load(f'{DRIVE}/probe_humor_vs_horror.pt').float().numpy()
sv_raw = torch.load(f'{DRIVE}/steering_vecs.pt')
sv = {k: v.to(device).to(torch.bfloat16) for k, v in sv_raw.items()}

with open(f'{DRIVE}/best_configs.json') as f:
    CONFIGS = json.load(f)
CONFIGS['humor']['layer'] = 18
CONFIGS['humor']['mult']  = 12.0

PROBE_FOR  = {'humor': w_hm, 'melancholy': w_hm, 'horror': w_hh}
PROBE_SIGN = {'humor': +1,   'melancholy': -1,   'horror': -1}

def coherence_score(text):
    words = text.strip().split()
    if len(words) < 3: return 0.0
    counts = Counter(words)
    if counts.most_common(1)[0][1] / len(words) > 0.4: return 0.0
    chars = [c for c in text.lower() if c.isalpha()]
    if len(chars) < 10: return 0.0
    cc  = Counter(chars)
    ent = -sum((c/len(chars))*math.log2(c/len(chars)) for c in cc.values())
    if ent < 3.5: return 0.0
    G = ['ss-','ses','sse','sss','thed','bsp','antml']
    def ok(w):
        w = w.strip('.,!?-()').lower()
        return len(w) < 2 or (any(c in 'aeiou' for c in w) and not any(p in w for p in G))
    return sum(ok(w) for w in words) / len(words)

def get_activation(text):
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > 512: tokens = tokens[:, -512:]
    hn = f'blocks.{PROBE_LAYER}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: hn in n, stop_at_layer=PROBE_LAYER+1)
    act = cache[hn][0,-1,:].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return act

def probe_score_fn(text, w, sign):
    return sign * float(get_activation(text) @ w)

def build_hook(emotion):
    cfg = CONFIGS[emotion]; svec = sv[emotion]
    layer = int(cfg['layer']); mult = float(cfg['mult'])
    def fn(resid_pre, hook):
        proj = (resid_pre @ svec).unsqueeze(-1) * svec
        return (resid_pre - proj) + svec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]

print(f'✅ Resume complete  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')

# ═══════════════════════════════════════════════════════════════════
# CELL 6: TEMPERATURE SWEEP
# ═══════════════════════════════════════════════════════════════════

# Load cell 9 (temp sweep) source directly from notebook cell
TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SAMPLE_N     = 100
PROBE_LAYER_LOCAL = 26

eval_sets = {
    'humor':      df_humor.sample(SAMPLE_N, random_state=2026).reset_index(drop=True),
    'melancholy': df_mel.sample(SAMPLE_N,   random_state=2026).reset_index(drop=True),
    'horror':     df_hor.sample(SAMPLE_N,   random_state=2026).reset_index(drop=True),
}

SWEEP_CKPT = f'{DRIVE}/temp_sweep_checkpoint.json'
SWEEP_CSV  = f'{DRIVE}/temp_sweep_results.csv'

if os.path.exists(SWEEP_CSV) and os.path.exists(SWEEP_CKPT):
    df_sweep = pd.read_csv(SWEEP_CSV)
    with open(SWEEP_CKPT) as f:
        done_keys = set(json.load(f)['done'])
    print(f'Temp sweep: loaded {len(df_sweep)} rows from checkpoint')
else:
    df_sweep  = pd.DataFrame()
    done_keys = set()

sweep_rows = df_sweep.to_dict('records') if len(df_sweep) else []

def get_three_scores(text):
    act = get_activation(text)
    return {
        'score_humor': float(act @ w_hm),
        'score_mel':   float(act @ (-w_hm)),
        'score_horror':float(act @ (-w_hh)),
    }

all_jobs = [(t, em) for t in TEMPERATURES for em in EMOTIONS]

with tqdm(total=len(all_jobs), desc='Overall') as pbar:
    for temp, emotion in all_jobs:
        key = f'{temp:.1f}_{emotion}'
        if key in done_keys:
            pbar.update(1); continue

        actual_t = 1e-10 if temp == 0.0 else temp
        hooks    = build_hook(emotion)
        w        = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
        df_e     = eval_sets[emotion]

        for _, row in df_e.iterrows():
            p = str(row['text'])[:120]
            base_out = model.generate(p, max_new_tokens=60, temperature=actual_t, verbose=False)
            base_ps  = probe_score_fn(base_out, w, sign)
            with model.hooks(fwd_hooks=hooks):
                ste_out = model.generate(p, max_new_tokens=60, temperature=actual_t, verbose=False)
            ste_gen = ste_out[len(p):].strip()
            ste_ps  = probe_score_fn(ste_out, w, sign)
            coh     = coherence_score(ste_gen)
            scores  = get_three_scores(ste_out)
            pred_em = max(scores, key=scores.get).replace('score_','').replace('mel','melancholy')
            sweep_rows.append({
                'temperature':    temp,
                'emotion_target': emotion,
                'prompt':         p[:60],
                'base_gen':       base_out[len(p):].strip()[:100],
                'ste_gen':        ste_gen[:100],
                'ste_coh':        float(coh),
                'is_coherent':    bool(coh >= 0.6),
                'ps_shift':       float(ste_ps - base_ps),
                'ps_success':     bool(ste_ps > base_ps),
                'both':           bool((ste_ps > base_ps) and (coh >= 0.6)),
                'score_humor':    scores['score_humor'],
                'score_mel':      scores['score_mel'],
                'score_horror':   scores['score_horror'],
                'pred_emotion':   pred_em,
                'pred_correct':   bool(pred_em == emotion),
            })

        done_keys.add(key)
        pd.DataFrame(sweep_rows).to_csv(SWEEP_CSV, index=False)
        with open(SWEEP_CKPT,'w') as f:
            json.dump({'done': list(done_keys)}, f)
        pbar.update(1)

df_sweep = pd.read_csv(SWEEP_CSV)

BEST_TEMP = max(
    TEMPERATURES,
    key=lambda t: np.mean([
        df_sweep[(df_sweep['temperature']==t)&(df_sweep['emotion_target']==em)]['both'].mean()
        for em in EMOTIONS
    ])
)
CONFIGS['best_temp'] = float(BEST_TEMP)
with open(f'{DRIVE}/best_configs.json','w') as f:
    json.dump(CONFIGS, f, indent=2)

df_sweep[df_sweep['temperature']==BEST_TEMP].to_csv(
    f'{DRIVE}/simplex_results_final.csv', index=False)

print(f'\nBEST_TEMP = {BEST_TEMP}  (avg_both = {np.mean([df_sweep[(df_sweep["temperature"]==BEST_TEMP)&(df_sweep["emotion_target"]==em)]["both"].mean() for em in EMOTIONS]):.3f})')
print('✅ Cell 6 complete')

# ═══════════════════════════════════════════════════════════════════
# CELL 7: MAIN EVAL AT T=0.7 + VECTOR COMPARISON
# ═══════════════════════════════════════════════════════════════════

EVAL_TEMP  = 0.7
df_results = df_sweep[df_sweep['temperature'] == EVAL_TEMP].copy()
df_results.to_csv(f'{DRIVE}/simplex_results_T07.csv', index=False)
print(f'Working at T={EVAL_TEMP}  ({len(df_results)} rows)')

# Confusion matrix
conf = np.zeros((3,3))
for i, te in enumerate(EMOTIONS):
    sub = df_results[df_results['emotion_target']==te]
    for j, pe in enumerate(EMOTIONS):
        conf[i,j] = (sub['pred_emotion']==pe).sum() / max(len(sub),1)

print(f'\nProbe confusion matrix (T={EVAL_TEMP}):')
print(f'  {"":12} {"→humor":>10} {"→mel":>10} {"→horror":>10}')
for i, em in enumerate(EMOTIONS):
    print(f'  {em:12} {conf[i,0]:>10.1%} {conf[i,1]:>10.1%} {conf[i,2]:>10.1%}')
print(f'  Avg diagonal: {np.diag(conf).mean():.1%}')

for emotion in EMOTIONS:
    sub = df_results[df_results['emotion_target']==emotion]
    print(f'\n  {emotion.upper()}: both={sub["both"].mean():.1%}  pred={sub["pred_correct"].mean():.1%}  coh={sub["is_coherent"].mean():.1%}  shift={sub["ps_shift"].mean():+.4f}')

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(conf, annot=True, fmt='.1%',
            xticklabels=['→humor','→mel','→horror'],
            yticklabels=['humor','mel','horror'],
            cmap='Blues', ax=ax, vmin=0, vmax=1)
ax.set_title(f'Probe Confusion Matrix  T={EVAL_TEMP}  avg_diag={np.diag(conf).mean():.1%}')
plt.tight_layout()
plt.savefig(f'{DRIVE}/probe_confusion_matrix_T07.png', dpi=150, bbox_inches='tight')
plt.show()

# Vector comparison
def _build_hook_vec(vec, emotion):
    cfg  = CONFIGS[emotion]; mult = float(cfg['mult']); layer = int(cfg['layer'])
    def fn(resid_pre, hook):
        proj = (resid_pre @ vec).unsqueeze(-1) * vec
        return (resid_pre - proj) + vec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]

VEC_N   = 50; actual_t = 1e-10 if EVAL_TEMP==0.0 else EVAL_TEMP
vec_sets = {}
for emotion in EMOTIONS:
    md   = sv[emotion].cpu().float()
    pw   = torch.tensor(PROBE_FOR[emotion] * PROBE_SIGN[emotion], dtype=torch.float32)
    pw   = pw / pw.norm().clamp(min=1e-8)
    rnd  = torch.randn_like(md); rnd = rnd / rnd.norm().clamp(min=1e-8)
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
    for vec_name, vec in vec_sets[emotion].items():
        print(f'  {emotion:12} {vec_name:12} ...', end='', flush=True)
        w = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
        hooks = _build_hook_vec(vec, emotion)
        results = []
        for prompt in eval_prompts_vc[emotion]:
            p = prompt[:120]
            base_ps = probe_score_fn(
                model.generate(p, max_new_tokens=60, temperature=actual_t, verbose=False), w, sign)
            with model.hooks(fwd_hooks=hooks):
                ste_out = model.generate(p, max_new_tokens=60, temperature=actual_t, verbose=False)
            ste_ps = probe_score_fn(ste_out, w, sign)
            coh    = coherence_score(ste_out[len(p):].strip())
            results.append({'both': bool((ste_ps>base_ps)and(coh>=0.6)),
                            'ps_shift': float(ste_ps-base_ps),
                            'vec': vec_name, 'emotion': emotion})
        both_r = np.mean([r['both'] for r in results])
        print(f'  both={both_r:.1%}')
        vc_rows.extend(results)

df_vc = pd.DataFrame(vc_rows)
df_vc.to_csv(f'{DRIVE}/vector_comparison.csv', index=False)

# Bonferroni stats
vec_names = ['mean_diff','probe_w','random']
n_comparisons = len(EMOTIONS) * len(list(combinations(vec_names,2)))
pval_results = []
for emotion in EMOTIONS:
    sub = df_vc[df_vc['emotion']==emotion]
    for v1, v2 in combinations(vec_names, 2):
        s1 = sub[sub['vec']==v1]['ps_shift'].values
        s2 = sub[sub['vec']==v2]['ps_shift'].values
        stat, p = scipy_stats.mannwhitneyu(s1, s2, alternative='two-sided')
        p_adj = min(p * n_comparisons, 1.0)
        sig   = '***' if p_adj<0.001 else ('**' if p_adj<0.01 else ('*' if p_adj<0.05 else 'ns'))
        pval_results.append({'emotion':emotion,'v1':v1,'v2':v2,
                             'U':float(stat),'p_raw':float(p),'p_bonf':float(p_adj),'sig':sig})
        print(f'  {emotion:12} {v1:12} vs {v2:12}  p_bonf={p_adj:.4f}  {sig}')

pd.DataFrame(pval_results).to_csv(f'{DRIVE}/vector_comparison_stats.csv', index=False)

cell7_summary = {
    'eval_temp': EVAL_TEMP,
    'avg_diagonal_T07': float(np.diag(conf).mean()),
    'per_emotion': {em: {
        'both': float(df_results[df_results['emotion_target']==em]['both'].mean()),
        'pred': float(df_results[df_results['emotion_target']==em]['pred_correct'].mean()),
    } for em in EMOTIONS},
}
with open(f'{DRIVE}/cell7_summary.json','w') as f:
    json.dump(cell7_summary, f, indent=2)

print(f'\n✅ Script 3 complete')
print(f'   Saved: temp_sweep_results.csv, simplex_results_T07.csv,')
print(f'          probe_confusion_matrix_T07.png, vector_comparison.csv,')
print(f'          vector_comparison_stats.csv, cell7_summary.json')
