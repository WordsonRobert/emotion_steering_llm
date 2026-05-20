# cell13_attention_head_ablation.py
#
# Ablates attention heads at a subset of layers: [0, 7, 14, 15, 18, 26].
# For each (layer, head): zeros out hook_z for that head and measures
# the drop in binary probe accuracy on n=100 balanced humor/mel texts.
# Threshold: drop > 2% → head is load-bearing ("critical").
# Checkpoint-resumable.
#
# Requires (from cell12):
#   model, N_LAYERS, D_MODEL, N_HEADS, PROBE_LAYER, DRIVE
#   df_humor, df_mel, CONFIGS, probe_weights
#
# Produces:
#   df_abl — ablation results DataFrame
#   cell9_summary.json  (critical_heads list — used by cell17 Exp D)
#   Saved: task19_ablation_results.csv, task19_ablation_results.json
#          task19_head_ablation.png, cell9_summary.json
#
# RESULT: 10 critical heads, all at L0. Top: L0H7 drop=0.060.
#         Layers 7, 14, 15, 18, 26: zero critical heads.
#
# Runtime: ~2 hrs on A100

import torch, numpy as np, pandas as pd, json, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

steer_layers    = list(set([CONFIGS[em]['layer'] for em in EMOTIONS]))
ablation_layers = sorted(set([0, N_LAYERS//4, N_LAYERS//2, PROBE_LAYER] + steer_layers))
print(f'Ablation layers: {ablation_layers}  ({len(ablation_layers)} layers)')
print(f'Heads per layer: {model.cfg.n_heads}')

N_ABL  = 100
rng    = np.random.default_rng(42)
n_each = N_ABL // 2
h_idx  = rng.choice(len(df_humor), size=n_each, replace=False)
m_idx  = rng.choice(len(df_mel),   size=n_each, replace=False)
abl_texts  = df_humor['text'].iloc[h_idx].tolist() + df_mel['text'].iloc[m_idx].tolist()
abl_labels = [1]*n_each + [0]*n_each

probe_weights = torch.load(f'{DRIVE}/probe_weights.pt', map_location='cpu')


def get_acts_at_layer(texts, layer, ablate_head=None, ablate_layer=None):
    acts = []
    hook_name = f'blocks.{layer}.hook_resid_post'
    for text in texts:
        tokens = model.to_tokens([text], prepend_bos=True)
        if tokens.shape[1] > 256: tokens = tokens[:, -256:]
        hooks = []
        if ablate_head is not None:
            h = ablate_head
            def make_zero_hook(head_idx):
                def fn(z, hook):
                    z_new = z.clone(); z_new[:, :, head_idx, :] = 0.0; return z_new
                return fn
            hooks = [(f'blocks.{ablate_layer}.attn.hook_z', make_zero_hook(h))]
        with torch.no_grad():
            if hooks:
                with model.hooks(fwd_hooks=hooks):
                    _, cache = model.run_with_cache(tokens, names_filter=lambda n: hook_name in n, stop_at_layer=layer+1)
            else:
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: hook_name in n, stop_at_layer=layer+1)
        act = cache[hook_name][0, -1, :].cpu().float().numpy()
        del cache; torch.cuda.empty_cache()
        acts.append(act)
    return np.array(acts)


def probe_accuracy(acts, labels, layer):
    w      = probe_weights[layer].float().numpy()
    scores = acts @ w
    preds  = (scores > 0).astype(int)
    return float((preds == np.array(labels)).mean())


CHECKPOINT = f'{DRIVE}/task19_ablation_results.json'
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f: results = json.load(f)
    print(f'Resumed: {len(results)} pairs done')
else:
    results = {}

n_heads = model.cfg.n_heads
for layer in tqdm(ablation_layers, desc='Ablation layers'):
    baseline_key = f'baseline_L{layer}'
    if baseline_key not in results:
        base_acts = get_acts_at_layer(abl_texts, layer)
        base_acc  = probe_accuracy(base_acts, abl_labels, layer)
        results[baseline_key] = float(base_acc)
        with open(CHECKPOINT, 'w') as f: json.dump(results, f)
        tqdm.write(f'  L{layer:2d} baseline acc = {base_acc:.3f}')
    else:
        base_acc = results[baseline_key]
        tqdm.write(f'  L{layer:2d} baseline acc = {base_acc:.3f} (cached)')

    for head_idx in tqdm(range(n_heads), desc=f'  L{layer} heads', leave=False):
        key = f'L{layer}_H{head_idx}'
        if key in results: continue
        abl_acts = get_acts_at_layer(abl_texts, layer, ablate_head=head_idx, ablate_layer=layer)
        abl_acc  = probe_accuracy(abl_acts, abl_labels, layer)
        drop     = base_acc - abl_acc
        results[key] = {'layer': layer, 'head': head_idx, 'baseline_acc': float(base_acc),
                        'ablated_acc': float(abl_acc), 'drop': float(drop), 'load_bearing': bool(drop > 0.02)}
    with open(CHECKPOINT, 'w') as f: json.dump(results, f)

print('\n✅ Ablation complete')

rows    = [v for k, v in results.items() if isinstance(v, dict)]
df_abl  = pd.DataFrame(rows)
df_abl.to_csv(f'{DRIVE}/task19_ablation_results.csv', index=False)

critical_heads = df_abl[df_abl['load_bearing']].sort_values('drop', ascending=False)
print(f'\nCritical heads (drop > 2%): {len(critical_heads)}')
print(critical_heads[['layer','head','baseline_acc','ablated_acc','drop']].to_string(index=False))

print(f'\nPer-layer critical head count:')
for layer in ablation_layers:
    sub = df_abl[df_abl['layer'] == layer]
    print(f'  L{layer:2d}: {sub["load_bearing"].sum():2d} critical heads  max_drop={sub["drop"].max():.3f}')

fig, axes = plt.subplots(1, len(ablation_layers), figsize=(4*len(ablation_layers), 5))
if len(ablation_layers) == 1: axes = [axes]
for idx, layer in enumerate(ablation_layers):
    ax  = axes[idx]; sub = df_abl[df_abl['layer'] == layer].sort_values('head')
    drops = sub['drop'].values
    colors = ['#C44E52' if d > 0.02 else '#4C72B0' for d in drops]
    ax.bar(range(n_heads), drops, color=colors, alpha=0.85)
    ax.axhline(0.02, color='k', lw=1, ls='--', label='2% threshold')
    ax.set_xlabel('Head'); ax.set_ylabel('Accuracy drop')
    ax.set_title(f'L{layer}  base={results[f"baseline_L{layer}"]:.2f}', fontsize=10)
    ax.set_xticks(range(0, n_heads, 4)); ax.grid(axis='y', alpha=0.3)
fig.suptitle('Task 19: Attention Head Ablation — Probe Accuracy Drop\nRed = load-bearing (drop > 2%)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE}/task19_head_ablation.png', dpi=130, bbox_inches='tight')
plt.show()
print('✅ task19_head_ablation.png saved')

cell9_summary = {
    'ablation_layers':  ablation_layers,
    'n_heads':          n_heads,
    'n_texts':          N_ABL,
    'threshold':        0.02,
    'n_critical_heads': int(len(critical_heads)),
    'critical_heads':   critical_heads[['layer','head','drop']].to_dict('records'),
}
with open(f'{DRIVE}/cell9_summary.json', 'w') as f:
    json.dump(cell9_summary, f, indent=2)
print('✅ cell9_summary.json saved')
print(f'\n━━━ CELL 13 COMPLETE ━━━')
