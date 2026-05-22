#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 8: EMOTIONAL ACTIVATION EXTRACTION (GPU)
# ═══════════════════════════════════════════════════════════════════
# Extension: Emotional Axes in Llama-3.2-3B
#
# Runs each emotional rollout through Llama-3.2-3B and extracts
# mean residual stream activations over RESPONSE TOKENS ONLY
# (not the system prompt or question tokens) at:
#   - L14 (middle layer)
#   - L26 (probe layer, N_LAYERS-2)
#
# Key detail: mean over response tokens only (not full sequence).
# Full-sequence mean causes system prompt to dominate, making all
# tone vectors nearly identical (cos ~+0.95). Response-only gives
# meaningful separation.
#
# Requires: Script 7 to have completed (reads emotional_rollouts.csv)
#
# Runtime:  ~15 min on A100
# Outputs:  EMOTIONAL_AXES/emotional_acts_L14.pt  (12390, 3072)
#           EMOTIONAL_AXES/emotional_acts_L26.pt  (12390, 3072)
#           EMOTIONAL_AXES/emotional_acts_meta.csv
# ═══════════════════════════════════════════════════════════════════

import os, json, torch
import numpy as np, pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

DRIVE_EA = '/content/drive/MyDrive/EMOTIONAL_AXES'
device   = 'cuda'

model = HookedTransformer.from_pretrained(
    'meta-llama/Llama-3.2-3B', dtype=torch.bfloat16, device=device)
model.eval()

N_LAYERS    = model.cfg.n_layers    # 28
D_MODEL     = model.cfg.d_model     # 3072
MID_LAYER   = N_LAYERS // 2         # 14
PROBE_LAYER = N_LAYERS - 2          # 26

print(f'N_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}  MID={MID_LAYER}  PROBE={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')

df = pd.read_csv(f'{DRIVE_EA}/emotional_rollouts.csv')
df['response']      = df['response'].fillna('').astype(str)
df['system_prompt'] = df['system_prompt'].fillna('').astype(str)
df['question']      = df['question'].fillna('').astype(str)
print(f'Loaded {len(df)} rollouts across {df["tone"].nunique()} tones')

# ── Checkpoint ────────────────────────────────────────────────────
CHECKPOINT = f'{DRIVE_EA}/phase2b_checkpoint.json'
ACTS_L14   = f'{DRIVE_EA}/emotional_acts_L14.pt'
ACTS_L26   = f'{DRIVE_EA}/emotional_acts_L26.pt'

if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f: ckpt = json.load(f)
    done_until = ckpt['done_until']
    acts_L14   = torch.load(ACTS_L14, map_location='cpu')
    acts_L26   = torch.load(ACTS_L26, map_location='cpu')
    print(f'Resumed from row {done_until}')
else:
    done_until = 0
    acts_L14   = torch.zeros(len(df), D_MODEL, dtype=torch.float32)
    acts_L26   = torch.zeros(len(df), D_MODEL, dtype=torch.float32)
    print('Starting fresh')

HOOK_L14 = f'blocks.{MID_LAYER}.hook_resid_post'
HOOK_L26 = f'blocks.{PROBE_LAYER}.hook_resid_post'

def extract_response_acts(system_prompt, question, response):
    """
    Extract mean residual stream activation over RESPONSE tokens only.
    Skips system prompt and question tokens to avoid contamination.
    """
    prefix      = f"{system_prompt}\n\n{question}\n\n"
    prefix_toks = model.to_tokens([prefix], prepend_bos=True)
    n_prefix    = prefix_toks.shape[1]

    full_text   = prefix + response
    full_toks   = model.to_tokens([full_text], prepend_bos=True)
    if full_toks.shape[1] > 512: full_toks = full_toks[:, :512]
    n_total     = full_toks.shape[1]

    resp_start  = min(n_prefix, n_total - 1)
    resp_end    = n_total
    if resp_start >= resp_end: resp_start = max(0, n_total - 10)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            full_toks,
            names_filter=lambda n: n in [HOOK_L14, HOOK_L26],
            stop_at_layer=PROBE_LAYER + 1,
        )
    a14 = cache[HOOK_L14][0, resp_start:resp_end].mean(0).cpu().float()
    a26 = cache[HOOK_L26][0, resp_start:resp_end].mean(0).cpu().float()
    del cache; torch.cuda.empty_cache()
    return a14, a26

SAVE_EVERY = 500

for idx in tqdm(range(done_until, len(df)), desc='Extracting'):
    row = df.iloc[idx]
    a14, a26 = extract_response_acts(row['system_prompt'], row['question'], row['response'])
    acts_L14[idx] = a14
    acts_L26[idx] = a26

    if (idx + 1) % SAVE_EVERY == 0:
        torch.save(acts_L14, ACTS_L14)
        torch.save(acts_L26, ACTS_L26)
        with open(CHECKPOINT,'w') as f: json.dump({'done_until': idx+1}, f)

torch.save(acts_L14, ACTS_L14)
torch.save(acts_L26, ACTS_L26)
with open(CHECKPOINT,'w') as f: json.dump({'done_until': len(df)}, f)

df[['tone','polarity','prompt_idx','question_idx','question']].to_csv(
    f'{DRIVE_EA}/emotional_acts_meta.csv', index=True)

# ── Sanity checks ─────────────────────────────────────────────────
print(f'\nSANITY CHECKS')
print(f'acts_L14: {acts_L14.shape}  norm={acts_L14.norm(dim=1).mean():.3f}')
print(f'acts_L26: {acts_L26.shape}  norm={acts_L26.norm(dim=1).mean():.3f}')
print(f'NaN L14: {acts_L14.isnan().any()}  NaN L26: {acts_L26.isnan().any()}')

# Check cosines — after centering should be meaningful
tones_check = [('humor','melancholy'),('humor','horror'),('melancholy','horror')]
global_mean = acts_L26[df['polarity']=='positive'].mean(0)
print('\nCentered cosines at L26 (positive prompts):')
for t1, t2 in tones_check:
    i1 = df[(df['tone']==t1)&(df['polarity']=='positive')].index
    i2 = df[(df['tone']==t2)&(df['polarity']=='positive')].index
    v1 = (acts_L26[i1].mean(0) - global_mean)
    v2 = (acts_L26[i2].mean(0) - global_mean)
    v1 = v1/(v1.norm()+1e-8); v2 = v2/(v2.norm()+1e-8)
    print(f'  cos({t1:12}, {t2:12}) = {float(v1@v2):+.4f}')

print(f'\n✅ Script 8 complete')
print(f'   Saved: emotional_acts_L14.pt, emotional_acts_L26.pt, emotional_acts_meta.csv')
