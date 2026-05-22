#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 2: PROBES + STEERING VECTORS + LAYER/MULT SWEEP
# ═══════════════════════════════════════════════════════════════════
# Covers notebook Cells 4-5
#
# Requires Script 1 to have completed (reads its outputs from Drive)
#
# What this does:
#   Cell 4: Three pairwise probes (humor vs mel, humor vs horror,
#            mel vs horror) + build mean-diff steering vectors
#   Cell 5: Layer sensitivity sweep + multiplier sweep → best_configs.json
#
# Runtime:  ~3.5 hrs on A100
# Outputs:  probe_humor_vs_melancholy.pt, probe_humor_vs_horror.pt,
#           probe_melancholy_vs_horror.pt, steering_vecs.pt,
#           best_configs.json, layer_heatmap.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

DRIVE   = '/content/drive/MyDrive/STEERING_EMNLP_2026'
DEVICE  = 'cuda'
EMOTIONS = ['humor', 'melancholy', 'horror']

# Load model
model = HookedTransformer.from_pretrained(
    'meta-llama/Llama-3.2-3B', dtype=torch.bfloat16, device=DEVICE)
model.eval()

N_LAYERS    = model.cfg.n_layers
D_MODEL     = model.cfg.d_model
PROBE_LAYER = N_LAYERS - 2
ALL_LAYERS  = list(range(N_LAYERS))

# Load data
df_humor = pd.read_csv(f'{DRIVE}/humor_master.csv')
df_mel   = pd.read_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor, df_mel, df_hor]:
    df['text'] = df['text'].fillna('').astype(str).str.strip()

humor_acts = torch.load(f'{DRIVE}/humor_acts.pt', map_location='cpu').float().numpy()
dark_acts  = torch.load(f'{DRIVE}/dark_acts.pt',  map_location='cpu').float().numpy()
probe_weights = torch.load(f'{DRIVE}/probe_weights.pt', map_location='cpu')

def norm_v(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

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
    act = cache[hn][0, -1, :].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return act

print(f'✅ Resume complete  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')

# ═══════════════════════════════════════════════════════════════════
# CELL 4: THREE PAIRWISE PROBES + STEERING VECTORS
# ═══════════════════════════════════════════════════════════════════

def extract_acts(texts):
    return np.array([get_activation(t) for t in tqdm(texts)], dtype=np.float32)

def normalize_for_probe(acts):
    scaler = StandardScaler()
    acts = scaler.fit_transform(acts)
    acts = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
    return acts, scaler

def train_probe(pos_acts, neg_acts, label):
    X = np.vstack([pos_acts, neg_acts])
    y = np.array([1]*len(pos_acts) + [0]*len(neg_acts))
    X_norm = normalize_for_probe(X)[0]
    clf = LogisticRegression(max_iter=1000, C=1.0)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X_norm, y, cv=cv).mean()
    clf.fit(X_norm, y)
    w = norm_v(clf.coef_[0].astype(np.float32))
    print(f'  {label}: CV acc = {acc:.4f}')
    return w, acc

print('Extracting per-emotion activations at PROBE_LAYER...')
acts_humor = extract_acts(df_humor['text'].tolist())
acts_mel   = extract_acts(df_mel['text'].tolist())
acts_hor   = extract_acts(df_hor['text'].tolist())

print('\nTraining pairwise probes...')
w_hm, acc_hm = train_probe(acts_humor, acts_mel,   'humor vs mel')
w_hh, acc_hh = train_probe(acts_humor, acts_hor,   'humor vs horror')
w_mh, acc_mh = train_probe(acts_mel,   acts_hor,   'mel vs horror')

torch.save(torch.tensor(w_hm), f'{DRIVE}/probe_humor_vs_melancholy.pt')
torch.save(torch.tensor(w_hh), f'{DRIVE}/probe_humor_vs_horror.pt')
torch.save(torch.tensor(w_mh), f'{DRIVE}/probe_melancholy_vs_horror.pt')

# Cosines
print('\nProbe orthogonality:')
for (n1,v1),(n2,v2) in [('hm','hh'),('hm','mh'),('hh','mh')]:
    pass
pairs = [('w_hm','w_hh',w_hm,w_hh), ('w_hm','w_mh',w_hm,w_mh), ('w_hh','w_mh',w_hh,w_mh)]
for n1,n2,v1,v2 in pairs:
    print(f'  cos({n1},{n2}) = {float(v1@v2):+.4f}')

# Build mean-diff steering vectors (whitened space)
def build_sv(pos_acts, neg_acts, label):
    X = np.vstack([pos_acts, neg_acts]).astype(np.float32)
    _, scaler = normalize_for_probe(X)
    pos_w = scaler.transform(pos_acts)
    neg_w = scaler.transform(neg_acts)
    pos_w = pos_w / (np.linalg.norm(pos_w, axis=1, keepdims=True) + 1e-8)
    neg_w = neg_w / (np.linalg.norm(neg_w, axis=1, keepdims=True) + 1e-8)
    sv = norm_v(pos_w.mean(0) - neg_w.mean(0)).astype(np.float32)
    print(f'  {label}: sv norm={np.linalg.norm(sv):.4f}')
    return sv

print('\nBuilding steering vectors...')
sv_humor = build_sv(acts_humor, np.vstack([acts_mel, acts_hor]), 'humor')
sv_mel   = build_sv(acts_mel,   np.vstack([acts_humor, acts_hor]), 'mel')
sv_hor   = build_sv(acts_hor,   np.vstack([acts_humor, acts_mel]), 'horror')

sv_raw = {
    'humor':     torch.tensor(sv_humor, dtype=torch.bfloat16),
    'melancholy':torch.tensor(sv_mel,   dtype=torch.bfloat16),
    'horror':    torch.tensor(sv_hor,   dtype=torch.bfloat16),
}
torch.save(sv_raw, f'{DRIVE}/steering_vecs.pt')
sv = {k: v.to(DEVICE) for k, v in sv_raw.items()}

PROBE_FOR  = {'humor': w_hm, 'melancholy': w_hm, 'horror': w_hh}
PROBE_SIGN = {'humor': +1,   'melancholy': -1,   'horror': -1}

print('\nSteering vector cosines:')
for (n1,v1),(n2,v2) in [('humor','mel'),('humor','horror'),('mel','horror')]:
    pass
sv_np = {k: v.cpu().float().numpy() for k, v in sv.items()}
for t1,t2 in [('humor','melancholy'),('humor','horror'),('melancholy','horror')]:
    print(f'  cos({t1},{t2}) = {float(norm_v(sv_np[t1])@norm_v(sv_np[t2])):+.4f}')

with open(f'{DRIVE}/cell4_summary.json','w') as f:
    json.dump({'probe_accs': {'hm':acc_hm,'hh':acc_hh,'mh':acc_mh}}, f, indent=2)
print('✅ Cell 4 complete')

# ═══════════════════════════════════════════════════════════════════
# CELL 5: LAYER SWEEP + MULTIPLIER SWEEP → best_configs.json
# ═══════════════════════════════════════════════════════════════════

def build_hook(layer, sv_vec, mult):
    def fn(resid_pre, hook):
        proj = (resid_pre @ sv_vec).unsqueeze(-1) * sv_vec
        return (resid_pre - proj) + sv_vec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]

def probe_score(text, w, sign):
    return sign * float(get_activation(text) @ w)

MID_MULT   = 9.0
SWEEP_N    = 30
LAYER_CKPT = f'{DRIVE}/cell5_layer_sweep_raw.json'

if os.path.exists(LAYER_CKPT):
    with open(LAYER_CKPT) as f:
        intervention_results = json.load(f)
    intervention_results = {em: {int(k):v for k,v in d.items()}
                            for em, d in intervention_results.items()}
    print(f'Layer sweep: loaded from checkpoint')
else:
    intervention_results = {em: {} for em in EMOTIONS}
    prompts = {
        'humor':      df_humor.sample(SWEEP_N, random_state=99)['text'].tolist(),
        'melancholy': df_mel.sample(SWEEP_N,   random_state=99)['text'].tolist(),
        'horror':     df_hor.sample(SWEEP_N,   random_state=99)['text'].tolist(),
    }
    for layer in tqdm(ALL_LAYERS, desc='Layer sweep'):
        for emotion in EMOTIONS:
            if layer in intervention_results[emotion]: continue
            w    = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
            hooks = build_hook(layer, sv[emotion], MID_MULT)
            shifts = []
            for prompt in prompts[emotion]:
                p = prompt[:120]
                base_ps = probe_score(
                    model.generate(p, max_new_tokens=30, temperature=0.7, verbose=False),
                    w, sign)
                with model.hooks(fwd_hooks=hooks):
                    ste_out = model.generate(p, max_new_tokens=30,
                                             temperature=0.7, verbose=False)
                shifts.append(probe_score(ste_out, w, sign) - base_ps)
            intervention_results[emotion][layer] = float(np.mean(shifts))
        with open(LAYER_CKPT,'w') as f:
            json.dump(intervention_results, f)

# Plot layer heatmap
fig, axes_plt = plt.subplots(1, 3, figsize=(15,4))
for idx, emotion in enumerate(EMOTIONS):
    ax = axes_plt[idx]
    vals = [intervention_results[emotion].get(l,0) for l in ALL_LAYERS]
    ax.bar(ALL_LAYERS, vals, color='#3498db', alpha=0.8)
    ax.set_title(f'{emotion.capitalize()}  (mult={MID_MULT})')
    ax.set_xlabel('Layer'); ax.set_ylabel('Mean probe shift')
    ax.axhline(0, color='k', lw=0.5)
fig.suptitle('Layer Sensitivity Sweep', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE}/layer_heatmap.png', dpi=130, bbox_inches='tight')
plt.show()

# Top-3 layers per emotion for multiplier sweep
top3 = {em: sorted(intervention_results[em], key=intervention_results[em].get,
                   reverse=True)[:3]
        for em in EMOTIONS}

MULTIPLIERS = [3.0, 6.0, 9.0, 12.0, 15.0]
MULT_N      = 15
MULT_CKPT   = f'{DRIVE}/cell5_mult_sweep_full.json'

if os.path.exists(MULT_CKPT):
    with open(MULT_CKPT) as f:
        mult_results = json.load(f)
    print('Mult sweep: loaded from checkpoint')
else:
    mult_results = {}
    for emotion in EMOTIONS:
        mult_results[emotion] = {}
        prompts_m = {
            'humor':      df_humor.sample(MULT_N, random_state=77)['text'].tolist(),
            'melancholy': df_mel.sample(MULT_N,   random_state=77)['text'].tolist(),
            'horror':     df_hor.sample(MULT_N,   random_state=77)['text'].tolist(),
        }[emotion]
        for layer in top3[emotion]:
            mult_results[emotion][str(layer)] = {}
            for mult in tqdm(MULTIPLIERS, desc=f'{emotion} L{layer}'):
                hooks = build_hook(layer, sv[emotion], mult)
                w     = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
                rows  = []
                for prompt in prompts_m:
                    p = prompt[:120]
                    base_ps = probe_score(
                        model.generate(p, max_new_tokens=30, temperature=0.7, verbose=False),
                        w, sign)
                    with model.hooks(fwd_hooks=hooks):
                        ste_out = model.generate(p, max_new_tokens=30,
                                                 temperature=0.7, verbose=False)
                    ste_ps  = probe_score(ste_out, w, sign)
                    gen_txt = ste_out[len(p):].strip()
                    coh     = coherence_score(gen_txt)
                    rows.append({'shift': ste_ps-base_ps, 'coh': coh,
                                 'both': (ste_ps>base_ps) and (coh>=0.6)})
                mult_results[emotion][str(layer)][str(mult)] = {
                    'both': float(np.mean([r['both'] for r in rows])),
                    'shift':float(np.mean([r['shift'] for r in rows])),
                }
        with open(MULT_CKPT,'w') as f:
            json.dump(mult_results, f)

# Select best (layer, mult) per emotion
CONFIGS = {}
for emotion in EMOTIONS:
    best_both, best_layer, best_mult = -1, None, None
    for layer_str, mults in mult_results[emotion].items():
        for mult_str, metrics in mults.items():
            if metrics['both'] > best_both:
                best_both  = metrics['both']
                best_layer = int(layer_str)
                best_mult  = float(mult_str)
    CONFIGS[emotion] = {'layer': best_layer, 'mult': best_mult, 'both': best_both}

# Override humor (grid search sometimes picks wrong layer)
CONFIGS['humor']['layer'] = 18
CONFIGS['humor']['mult']  = 12.0
CONFIGS['best_temp']      = 1.0   # filled by Cell 6 (temp sweep)

with open(f'{DRIVE}/best_configs.json','w') as f:
    json.dump(CONFIGS, f, indent=2)

print('\nFinal CONFIGS:')
for em in EMOTIONS:
    print(f'  {em}: L{CONFIGS[em]["layer"]} × {CONFIGS[em]["mult"]}  both={CONFIGS[em]["both"]:.3f}')

print('✅ Script 2 complete')
print('   Saved: probe_*.pt, steering_vecs.pt, best_configs.json, layer_heatmap.png')
