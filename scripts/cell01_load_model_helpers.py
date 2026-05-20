# cell01_load_model_helpers.py
#
# Loads Llama-3.2-3B via TransformerLens with VRAM-saving unembed offload.
# Defines ALL shared helpers used by every downstream cell.
#
# Requires:
#   DRIVE (from cell00)
#   HuggingFace token already set
#
# Produces (in memory):
#   model, DEVICE
#   N_LAYERS, D_MODEL, N_HEADS, ALL_LAYERS, PROBE_LAYER
#   EMOTIONS, SEEDS
#   norm_v, get_activation_at_layer, get_activation
#   coherence_score, type_token_ratio
#   probe_score
#   make_proj_inject_hook, make_additive_hook
#   generate
#   normalize_for_probe
#
# Runtime: ~4 min

import os, gc, math, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from contextlib import nullcontext
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer

# ── Device ────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE != 'cuda':
    print('⚠️  WARNING: No GPU detected. This pipeline requires an A100.')

# ── Load model ────────────────────────────────────────────────────
MODEL_NAME = 'meta-llama/Llama-3.2-3B'
print(f'Loading {MODEL_NAME}...')
gc.collect(); torch.cuda.empty_cache()

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device=DEVICE,
)
model.eval()

# ── Unembed CPU offload ───────────────────────────────────────────
# W_U is [D_MODEL, vocab_size] ~750 MB in bfloat16.
# Moving it to CPU saves VRAM without breaking generation.
model.unembed.W_U = nn.Parameter(model.unembed.W_U.cpu())
if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None:
    model.unembed.b_U = nn.Parameter(model.unembed.b_U.cpu())

def _patched_unembed_forward(self, residual):
    W = self.W_U.to(residual.device)
    b = (self.b_U.to(residual.device)
         if hasattr(self, 'b_U') and self.b_U is not None else None)
    return self.hook_out(F.linear(residual, W.T.contiguous(), b))

model.unembed.__class__.forward = _patched_unembed_forward
gc.collect(); torch.cuda.empty_cache()

# ── Architecture constants — ALL from model.cfg ───────────────────
N_LAYERS    = model.cfg.n_layers          # 28 for Llama-3.2-3B
D_MODEL     = model.cfg.d_model           # 3072
N_HEADS     = model.cfg.n_heads           # 24
ALL_LAYERS  = list(range(N_LAYERS))
PROBE_LAYER = N_LAYERS - 2               # 26

# ── Fixed experiment constants ────────────────────────────────────
EMOTIONS = ['humor', 'melancholy', 'horror']
SEEDS    = {'train': 42, 'eval': 2026, 'sweep': 99}

print(f'✅ Model loaded')
print(f'   N_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}  N_HEADS={N_HEADS}')
print(f'   PROBE_LAYER={PROBE_LAYER}')
print(f'   VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB')


# ════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════════════════════════════

def norm_v(v: torch.Tensor) -> torch.Tensor:
    """Safe L2 normalise. Works on CPU or GPU, any dtype."""
    v = v.float()
    return v / v.norm().clamp(min=1e-8)


def get_activation_at_layer(
    text: str,
    layer: int,
    max_len: int = 512,
) -> torch.Tensor:
    """
    Extract last-token resid_post at `layer`.
    Returns float32 CPU tensor of shape [D_MODEL].
    Uses stop_at_layer to avoid running the full forward pass.
    """
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > max_len:
        tokens = tokens[:, -max_len:]
    hook_name = f'blocks.{layer}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: hook_name in n,
            stop_at_layer=layer + 1,
        )
    act = cache[hook_name][0, -1, :].cpu().float()
    del cache; torch.cuda.empty_cache()
    return act


def get_activation(text: str) -> torch.Tensor:
    """Shorthand: extract at PROBE_LAYER."""
    return get_activation_at_layer(text, PROBE_LAYER)


def coherence_score(text: str) -> float:
    """
    TTR-based coherence. Returns 0.0 for degenerate output.
    Threshold >= 0.6 is the coherence boundary used everywhere.
    """
    words = text.strip().split()
    if len(words) < 3:
        return 0.0
    counts = Counter(words)
    if counts.most_common(1)[0][1] / len(words) > 0.4:
        return 0.0
    chars = [c for c in text.lower() if c.isalpha()]
    if len(chars) < 10:
        return 0.0
    char_counts = Counter(chars)
    entropy = -sum(
        (c / len(chars)) * math.log2(c / len(chars))
        for c in char_counts.values()
    )
    if entropy < 3.5:
        return 0.0
    GIBBERISH_PATTERNS = ['ss-', 'ses', 'sse', 'sss', 'thed', 'bsp', 'antml']
    def is_real_word(w):
        w = w.strip('.,!?-()[]"\'';:').lower()
        if len(w) < 2:
            return True
        has_vowel = any(c in 'aeiou' for c in w)
        has_gibberish = any(p in w for p in GIBBERISH_PATTERNS)
        return has_vowel and not has_gibberish
    return sum(is_real_word(w) for w in words) / len(words)


def type_token_ratio(text: str) -> float:
    words = text.strip().lower().split()
    return len(set(words)) / len(words) if words else 0.0


def probe_score(text: str, w: np.ndarray, sign: int) -> float:
    """Score `text` with probe vector `w` at PROBE_LAYER."""
    act = get_activation(text).numpy()
    return sign * float(act @ w)


def make_proj_inject_hook(sv_vec: torch.Tensor, mult: float):
    """
    Projection-injection hook (paper's Algorithm 1):
      1. Remove existing component along sv_vec
      2. Add sv_vec * mult
    sv_vec must be on DEVICE in bfloat16.
    """
    v = sv_vec
    def fn(resid_pre, hook):
        proj = (resid_pre @ v).unsqueeze(-1) * v
        return (resid_pre - proj) + v * mult
    return fn


def make_additive_hook(sv_vec: torch.Tensor, mult: float):
    """
    Additive hook (baseline — no projection removal):
      resid_pre += sv_vec * mult
    Used in Task 2 ablation.
    """
    v = sv_vec
    def fn(resid_pre, hook):
        return resid_pre + v * mult
    return fn


def generate(
    prompt: str,
    hooks=None,
    temperature: float = 0.7,
    max_new_tokens: int = 60,
) -> str:
    """
    Unified generation wrapper. Returns full string (prompt + generation).
    temperature=0.0 uses greedy decoding.
    """
    actual_temp = 1e-10 if temperature == 0.0 else temperature
    ctx = model.hooks(fwd_hooks=hooks) if hooks else nullcontext()
    with torch.no_grad(), ctx:
        out = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=actual_temp,
            verbose=False,
        )
    torch.cuda.empty_cache()
    return out


def normalize_for_probe(X: np.ndarray):
    """
    Z-score then row-normalize.
    Returns (X_normalized, mean, std).
    """
    mean = X.mean(0)
    std  = X.std(0) + 1e-8
    Xn   = (X - mean) / std
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return Xn / norms, mean.astype(np.float32), std.astype(np.float32)


print('\n✅ Cell 1 complete — all helpers defined')
