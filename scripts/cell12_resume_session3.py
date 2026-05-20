# cell12_resume_session3.py
#
# Resume block for session 3 (attention head ablation and beyond).
# Reloads model, CSVs, configs, and probe weights.
#
# Requires on disk: all .pt files, best_configs.json, master CSVs
# Produces: everything needed for cell13 onward

import os, json, math, torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformer_lens import HookedTransformer

DRIVE    = '/content/drive/MyDrive/STEERING_EMNLP_2026'
device   = 'cuda'
EMOTIONS = ['humor', 'melancholy', 'horror']

with open(f'{DRIVE}/best_configs.json') as f:
    CONFIGS = json.load(f)
CONFIGS['humor']['layer'] = 18
CONFIGS['humor']['mult']  = 12.0
print(CONFIGS)

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

probe_weights = torch.load(f'{DRIVE}/probe_weights.pt', map_location='cpu')
print(f'✅ Ready  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'   humor={len(df_humor)}  mel={len(df_mel)}  horror={len(df_hor)}')
print(f'   probe_weights layers: {len(probe_weights)}')
