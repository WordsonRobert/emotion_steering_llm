# cell07_resume_session2.py
#
# Resume block for the second Colab session (temperature sweep onward).
# Run if runtime was cut after cell06. Reloads model and all artifacts.
# Does NOT apply the unembed offload patch (matching original notebook behavior).
#
# Requires on disk: all .pt files, best_configs.json, master CSVs
# Produces: everything needed for cell08 onward

import os, gc, json, math, torch, numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from contextlib import nullcontext
from tqdm import tqdm
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer

from google.colab import drive
drive.mount('/content/drive')

DRIVE    = '/content/drive/MyDrive/STEERING_EMNLP_2026'
device   = 'cuda'
EMOTIONS = ['humor', 'melancholy', 'horror']

model = HookedTransformer.from_pretrained(
    'meta-llama/Llama-3.2-3B', dtype=torch.bfloat16, device=device)
model.eval()
# NOTE: No offload patch here — matching original notebook for this session.

N_LAYERS    = model.cfg.n_layers
D_MODEL     = model.cfg.d_model
PROBE_LAYER = N_LAYERS - 2

df_humor = pd.read_csv(f'{DRIVE}/humor_master.csv')
df_mel   = pd.read_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor, df_mel, df_hor]:
    df['text'] = df['text'].fillna('').astype(str).str.strip()

print(f'✅ VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'   humor={len(df_humor)}  mel={len(df_mel)}  horror={len(df_hor)}')
