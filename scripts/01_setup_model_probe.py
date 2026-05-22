#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 1: SETUP + MODEL + DATA + ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════
# Covers notebook Cells 0-3
#
# What this does:
#   0. Install packages, mount Drive, HF login
#   1. Load Llama-3.2-3B via TransformerLens + define all shared helpers
#   2. Load & validate humor/melancholy/horror master CSVs
#   3. Extract activations at all 28 layers + train binary probe
#      → Saves humor_acts.pt, dark_acts.pt, probe_weights.pt
#      → Saves layer_probe_accuracy.png
#
# Runtime:  ~50 min on A100
# Outputs:  humor_acts.pt, dark_acts.pt, probe_weights.pt,
#           layer_probe_accuracy.png
# ═══════════════════════════════════════════════════════════════════

# ── Cell 0: Install + Mount + Discover ───────────────────────────
import subprocess, sys

def _pip(*args):
    subprocess.check_call(
        [sys.executable, '-m', 'pip'] + list(args),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

print('Installing packages...')
_pip('install', 'numpy==2.0.2', '--force-reinstall', '-q')
_pip('install', 'transformer_lens', '--no-deps', '-q')
_pip('install',
     'better-abc==0.0.3',
     'transformers-stream-generator==0.0.5',
     'beartype==0.14.1',
     'huggingface-hub>=0.23.2,<1.0',
     'einops', 'jaxtyping', 'fancy-einsum', '-q')
_pip('install',
     'scikit-learn', 'matplotlib', 'seaborn',
     'scipy', 'datasets', 'transformers', '-q')
print('✅ Packages installed')

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os

REQUIRED_CSVS = ['humor_master.csv', 'melancholy_master.csv', 'horror_master.csv']
SEARCH_ROOTS  = ['/content/drive/MyDrive', '/content/drive/Shareddrives']

def _has_csvs(path):
    return all(os.path.exists(os.path.join(path, f)) for f in REQUIRED_CSVS)

DRIVE = None
for root in SEARCH_ROOTS:
    if not os.path.exists(root): continue
    candidate = os.path.join(root, 'STEERING_EMNLP_2026')
    if os.path.exists(candidate) and _has_csvs(candidate):
        DRIVE = candidate; break

if DRIVE is None:
    for root in SEARCH_ROOTS:
        if not os.path.exists(root): continue
        try:
            for entry in os.scandir(root):
                if entry.is_dir() and _has_csvs(entry.path):
                    DRIVE = entry.path; break
        except PermissionError: pass
        if DRIVE: break

if DRIVE is None:
    print('⚠️  Could not auto-discover folder.')
    DRIVE = input('Paste full path to folder with master CSVs: ').strip().rstrip('/')
    assert _has_csvs(DRIVE), f'Missing CSVs in {DRIVE}'

print(f'✅ DRIVE = {DRIVE}')

HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception: pass

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print('✅ HuggingFace login successful')
else:
    print('⚠️  HF_TOKEN not found. Add it in Tools → Secrets → HF_TOKEN')

# ── Cell 1: Load Model + All Shared Helpers ───────────────────────
import gc, json, math, torch
import numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['humor', 'melancholy', 'horror']
SEEDS    = {'humor': 42, 'dark': 43}

print(f'Loading Llama-3.2-3B on {DEVICE}...')
model = HookedTransformer.from_pretrained(
    'meta-llama/Llama-3.2-3B',
    dtype=torch.bfloat16,
    device=DEVICE,
)
model.eval()

N_LAYERS    = model.cfg.n_layers     # 28
D_MODEL     = model.cfg.d_model      # 3072
N_HEADS     = model.cfg.n_heads      # 24
PROBE_LAYER = N_LAYERS - 2           # 26
ALL_LAYERS  = list(range(N_LAYERS))

print(f'N_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}  N_HEADS={N_HEADS}  PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')

# Shared helpers
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

def get_activation_at_layer(text, layer):
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > 512: tokens = tokens[:, -512:]
    hn = f'blocks.{layer}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: hn in n, stop_at_layer=layer+1)
    act = cache[hn][0, -1, :].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return act

def get_activation(text):
    return get_activation_at_layer(text, PROBE_LAYER)

def probe_score(text, w, sign):
    return sign * float(get_activation(text) @ w)

def norm_v(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def make_proj_inject_hook(sv_vec, mult):
    def fn(resid_pre, hook):
        proj = (resid_pre @ sv_vec).unsqueeze(-1) * sv_vec
        return (resid_pre - proj) + sv_vec * mult
    return fn

def make_additive_hook(sv_vec, mult):
    def fn(resid_pre, hook):
        return resid_pre + sv_vec * mult
    return fn

print('✅ Cell 1 complete — model + helpers ready')

# ── Cell 2: Load & Validate CSVs ─────────────────────────────────
def _load_csv(path):
    df = pd.read_csv(path)
    text_col = next((c for c in df.columns
                     if c.lower() in ['text','content','post','body']), df.columns[0])
    df = df.rename(columns={text_col: 'text'})
    df['text'] = df['text'].fillna('').astype(str).str.strip()
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    return df

df_humor = _load_csv(f'{DRIVE}/humor_master.csv')
df_mel   = _load_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = _load_csv(f'{DRIVE}/horror_master.csv')

dark_texts = df_mel['text'].tolist() + df_hor['text'].tolist()

print(f'humor={len(df_humor)}  mel={len(df_mel)}  horror={len(df_hor)}')
print(f'dark_texts={len(dark_texts)}')
print('✅ Cell 2 complete')

# ── Cell 3: Activation Extraction + Binary Probe All Layers ───────
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

PROBE_N = min(300, len(df_humor), len(dark_texts))

# Part A: Extract at PROBE_LAYER
def _extract_acts(texts, label):
    path = f'{DRIVE}/{label}_acts.pt'
    if os.path.exists(path):
        print(f'  {label}: loading cached {path}')
        return torch.load(path, map_location='cpu').float().numpy()
    acts = []
    for text in tqdm(texts, desc=f'  {label}'):
        acts.append(get_activation(text))
    arr = np.array(acts, dtype=np.float32)
    torch.save(torch.tensor(arr), path)
    return arr

print('Extracting humor activations...')
humor_acts = _extract_acts(df_humor['text'].tolist(), 'humor')
print('Extracting dark activations...')
dark_acts  = _extract_acts(dark_texts, 'dark')

# Part B: Per-layer probe
print('\nTraining binary probe at every layer...')
layer_probe_accs = {}
probe_weights    = {}

rng    = np.random.default_rng(42)
h_idx  = rng.choice(len(humor_acts), size=PROBE_N, replace=False)
d_idx  = rng.choice(len(dark_acts),  size=PROBE_N, replace=False)

for layer in tqdm(ALL_LAYERS, desc='Probe layers'):
    h_sub, d_sub = [], []
    for text in df_humor['text'].iloc[h_idx].tolist():
        h_sub.append(get_activation_at_layer(text, layer))
    for text in dark_texts[:PROBE_N]:
        d_sub.append(get_activation_at_layer(text, layer))

    X = np.vstack([h_sub, d_sub]).astype(np.float32)
    y = np.array([1]*PROBE_N + [0]*PROBE_N)

    scaler = StandardScaler(); X = scaler.fit_transform(X)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv).mean()
    layer_probe_accs[layer] = float(acc)

    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float32)
    probe_weights[layer] = norm_v(w)

torch.save(probe_weights, f'{DRIVE}/probe_weights.pt')

# Part C: Plot
fig, ax = plt.subplots(figsize=(12, 4))
layers  = list(layer_probe_accs.keys())
accs    = [layer_probe_accs[l] for l in layers]
colors  = ['#e74c3c' if l == PROBE_LAYER else '#3498db' for l in layers]
ax.bar(layers, accs, color=colors, alpha=0.85)
ax.axhline(0.5, color='gray', ls='--', lw=1, label='Chance')
ax.set_xlabel('Layer'); ax.set_ylabel('CV Accuracy')
ax.set_title('Binary Probe Accuracy per Layer (humor vs dark)')
ax.set_ylim(0.4, 1.05); ax.legend()
plt.tight_layout()
plt.savefig(f'{DRIVE}/layer_probe_accuracy.png', dpi=130, bbox_inches='tight')
plt.show()
print(f'Peak accuracy: L{max(layer_probe_accs, key=layer_probe_accs.get)} = '
      f'{max(layer_probe_accs.values()):.4f}')
print(f'✅ Script 1 complete')
print(f'   Saved: humor_acts.pt, dark_acts.pt, probe_weights.pt, layer_probe_accuracy.png')
