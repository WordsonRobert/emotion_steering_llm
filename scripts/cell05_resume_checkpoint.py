# cell05_resume_checkpoint.py
#
# Resume block for session continuations after cell04.
# Run this instead of cells 0–4 if the Colab runtime was cut.
# Reloads model, CSVs, saved .pt artifacts, and intervention_results checkpoint.
#
# Requires on disk (DRIVE):
#   humor_master.csv, melancholy_master.csv, horror_master.csv
#   probe_humor_vs_melancholy.pt, probe_humor_vs_horror.pt
#   steering_vecs.pt
#   cell5_layer_sweep_raw.json   (if layer sweep was already partially run)
#
# Produces (in memory):
#   Everything cells 0–4 produce, plus intervention_results from checkpoint

import os, gc, json, math, torch, numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from contextlib import nullcontext
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from transformer_lens import HookedTransformer

DRIVE      = '/content/drive/MyDrive/STEERING_EMNLP_2026'
DEVICE     = 'cuda'
EMOTIONS   = ['humor', 'melancholy', 'horror']
SEEDS      = {'train': 42, 'eval': 2026, 'sweep': 99}
MODEL_NAME = 'meta-llama/Llama-3.2-3B'

model = HookedTransformer.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device=DEVICE)
model.eval()
model.unembed.W_U = nn.Parameter(model.unembed.W_U.cpu())
if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None:
    model.unembed.b_U = nn.Parameter(model.unembed.b_U.cpu())

def _patched_unembed_forward(self, residual):
    W = self.W_U.to(residual.device)
    b = (self.b_U.to(residual.device)
         if hasattr(self, 'b_U') and self.b_U is not None else None)
    return self.hook_out(F.linear(residual, W.T.contiguous(), b))

model.unembed.__class__.forward = _patched_unembed_forward

N_LAYERS    = model.cfg.n_layers
D_MODEL     = model.cfg.d_model
N_HEADS     = model.cfg.n_heads
ALL_LAYERS  = list(range(N_LAYERS))
PROBE_LAYER = N_LAYERS - 2

df_humor = pd.read_csv(f'{DRIVE}/humor_master.csv')
df_mel   = pd.read_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor, df_mel, df_hor]:
    df['text'] = df['text'].fillna('').astype(str).str.strip()

humor_acts    = torch.load(f'{DRIVE}/humor_acts.pt',    map_location='cpu')
dark_acts     = torch.load(f'{DRIVE}/dark_acts.pt',     map_location='cpu')
probe_weights = torch.load(f'{DRIVE}/probe_weights.pt', map_location='cpu')

w_hm   = torch.load(f'{DRIVE}/probe_humor_vs_melancholy.pt', map_location='cpu').float().numpy()
w_hh   = torch.load(f'{DRIVE}/probe_humor_vs_horror.pt',     map_location='cpu').float().numpy()
sv_raw = torch.load(f'{DRIVE}/steering_vecs.pt', map_location='cpu')
sv     = {k: v.to(DEVICE).to(torch.bfloat16) for k, v in sv_raw.items()}

PROBE_FOR  = {'humor': w_hm, 'melancholy': w_hm, 'horror': w_hh}
PROBE_SIGN = {'humor': +1,   'melancholy': -1,   'horror': -1}


def norm_v(v): v = v.float(); return v / v.norm().clamp(min=1e-8)

def normalize_for_probe(X):
    mean = X.mean(0); std = X.std(0) + 1e-8; Xn = (X - mean) / std
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return Xn / norms, mean.astype(np.float32), std.astype(np.float32)

def get_activation_at_layer(text, layer, max_len=512):
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > max_len: tokens = tokens[:, -max_len:]
    hn = f'blocks.{layer}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: hn in n, stop_at_layer=layer+1)
    act = cache[hn][0, -1, :].cpu().float()
    del cache; torch.cuda.empty_cache()
    return act

def get_activation(text): return get_activation_at_layer(text, PROBE_LAYER)

def coherence_score(text):
    words = text.strip().split()
    if len(words) < 3: return 0.0
    counts = Counter(words)
    if counts.most_common(1)[0][1] / len(words) > 0.4: return 0.0
    chars = [c for c in text.lower() if c.isalpha()]
    if len(chars) < 10: return 0.0
    cc = Counter(chars)
    ent = -sum((c/len(chars))*math.log2(c/len(chars)) for c in cc.values())
    if ent < 3.5: return 0.0
    G = ['ss-','ses','sse','sss','thed','bsp','antml']
    def ok(w):
        w = w.strip('.,!?-()').lower()
        return len(w) < 2 or (any(c in 'aeiou' for c in w) and not any(p in w for p in G))
    return sum(ok(w) for w in words) / len(words)

def make_proj_inject_hook(sv_vec, mult):
    v = sv_vec
    def fn(resid_pre, hook):
        proj = (resid_pre @ v).unsqueeze(-1) * v
        return (resid_pre - proj) + v * mult
    return fn

def generate(prompt, hooks=None, temperature=0.7, max_new_tokens=60):
    actual = 1e-10 if temperature == 0.0 else temperature
    ctx = model.hooks(fwd_hooks=hooks) if hooks else nullcontext()
    with torch.no_grad(), ctx:
        out = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=actual, verbose=False)
    torch.cuda.empty_cache()
    return out

# Load layer sweep checkpoint if it exists
SWEEP_CHECKPOINT = f'{DRIVE}/cell5_layer_sweep_raw.json'
if os.path.exists(SWEEP_CHECKPOINT):
    with open(SWEEP_CHECKPOINT) as f:
        raw = json.load(f)
    intervention_results = {em: {int(l): v for l, v in d.items()} for em, d in raw.items()}
    n_done = sum(len(v) for v in intervention_results.values())
    print(f'  intervention_results: {n_done} layer results loaded from checkpoint')
else:
    intervention_results = {em: {} for em in EMOTIONS}
    print('  No layer sweep checkpoint found. Starting fresh.')

print('✅ Resume complete')
print(f'   N_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}  PROBE_LAYER={PROBE_LAYER}')
print(f'   humor={len(df_humor)}  mel={len(df_mel)}  horror={len(df_hor)}')
