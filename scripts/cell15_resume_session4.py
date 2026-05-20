# cell15_resume_session4.py
#
# Resume block for session 4 (Tasks 2+6 and ablation experiments).
# Reloads model, CSVs, probe weights, steering vecs, and all configs.
#
# Requires on disk: all .pt files, best_configs.json, master CSVs
# Produces: everything needed for cell16 onward

import os, gc, json, math, torch, numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer

from google.colab import drive
drive.mount('/content/drive')

DRIVE    = '/content/drive/MyDrive/STEERING_EMNLP_2026'
device   = 'cuda'
EMOTIONS = ['humor', 'melancholy', 'horror']

with open(f'{DRIVE}/best_configs.json') as f:
    CONFIGS = json.load(f)
CONFIGS['humor']['layer'] = 18
CONFIGS['humor']['mult']  = 12.0

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

w_hm   = torch.load(f'{DRIVE}/probe_humor_vs_melancholy.pt').float().numpy()
w_hh   = torch.load(f'{DRIVE}/probe_humor_vs_horror.pt').float().numpy()
sv_raw = torch.load(f'{DRIVE}/steering_vecs.pt')
sv     = {k: v.to(device).to(torch.bfloat16) for k, v in sv_raw.items()}

PROBE_FOR  = {'humor': w_hm, 'melancholy': w_hm, 'horror': w_hh}
PROBE_SIGN = {'humor': +1,   'melancholy': -1,   'horror': -1}

def coherence_score(text):
    words = text.strip().split()
    if len(words) < 3: return 0.0
    counts = Counter(words)
    if counts.most_common(1)[0][1] / len(words) > 0.4: return 0.0
    chars = [c for c in text.lower() if c.isalpha()]
    if len(chars) < 10: return 0.0
    cc = Counter(chars); ent = -sum((c/len(chars))*math.log2(c/len(chars)) for c in cc.values())
    if ent < 3.5: return 0.0
    G = ['ss-','ses','sse','sss','thed','bsp','antml']
    def ok(w):
        w = w.strip('.,!?-()').lower()
        return len(w) < 2 or (any(c in 'aeiou' for c in w) and not any(p in w for p in G))
    return sum(ok(w) for w in words) / len(words)

def probe_score(text, w, sign):
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > 512: tokens = tokens[:, -512:]
    hn = f'blocks.{PROBE_LAYER}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: hn in n, stop_at_layer=PROBE_LAYER+1)
    act = cache[hn][0, -1, :].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return sign * float(act @ w)

def build_hook(emotion):
    cfg = CONFIGS[emotion]; svec = sv[emotion]; layer = cfg['layer']; mult = cfg['mult']
    def fn(resid_pre, hook):
        proj = (resid_pre @ svec).unsqueeze(-1) * svec
        return (resid_pre - proj) + svec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]

print(f'✅ VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'   humor={len(df_humor)}  mel={len(df_mel)}  horror={len(df_hor)}')
