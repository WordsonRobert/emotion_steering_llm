# cell06_layer_mult_sweep.py
#
# Module A: Layer sensitivity sweep — runs steering at mult=9.0 across all 28 layers,
#           n=30 samples per emotion. Checkpoint-resumable (saves after every layer).
# Module B: Multiplier sweep on top-3 layers per emotion,
#           mults [3.0, 6.0, 9.0, 12.0, 15.0], n=15 per cell.
#           Best config = argmax both-rate, tie-broken by mean probe shift.
# Plots per-emotion layer heatmaps.
#
# Requires (from cell05 or cells 1–4):
#   model, DEVICE, N_LAYERS, D_MODEL, PROBE_LAYER, ALL_LAYERS
#   EMOTIONS, SEEDS, df_humor, df_mel, df_hor
#   sv, PROBE_FOR, PROBE_SIGN
#   make_proj_inject_hook, generate, coherence_score
#
# Produces (in memory + saved to DRIVE):
#   intervention_results — {emotion: {layer: mean_shift}}
#   CONFIGS              — {emotion: {layer, mult, both}}
#   Saved: best_configs.json, layer_heatmap.png,
#          cell5_layer_sweep_raw.json, cell5_mult_sweep_full.json
#
# RESULT: humor L18×12.0  melancholy L18×3.0  horror L15×9.0
#
# Runtime: ~3 hrs on A100

import torch, numpy as np, json, os
import matplotlib.pyplot as plt
from tqdm import tqdm

MID_MULT      = 9.0
MULTIPLIERS   = [3.0, 6.0, 9.0, 12.0, 15.0]
SAMPLE_N_A    = 30
SAMPLE_N_B    = 15
COH_THRESHOLD = 0.6

samples = {
    'humor':      df_humor.sample(SAMPLE_N_A, random_state=SEEDS['sweep'])['text'].tolist(),
    'melancholy': df_mel.sample(SAMPLE_N_A,   random_state=SEEDS['sweep'])['text'].tolist(),
    'horror':     df_hor.sample(SAMPLE_N_A,   random_state=SEEDS['sweep'])['text'].tolist(),
}
samples_b = {em: texts[:SAMPLE_N_B] for em, texts in samples.items()}


def _score_text(text, w, sign):
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > 512: tokens = tokens[:, -512:]
    hook_name = f'blocks.{PROBE_LAYER}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: hook_name in n, stop_at_layer=PROBE_LAYER+1)
    act = cache[hook_name][0, -1, :].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return sign * float(act @ w)


def _probe_shift(text, w, sign, interv_layer, sv_vec):
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > 512: tokens = tokens[:, -512:]
    hook_name = f'blocks.{PROBE_LAYER}.hook_resid_post'
    with torch.no_grad():
        _, cb = model.run_with_cache(tokens, names_filter=lambda n: hook_name in n, stop_at_layer=PROBE_LAYER+1)
    base_act = cb[hook_name][0, -1, :].cpu().float().numpy()
    del cb; torch.cuda.empty_cache()
    hook_fn = make_proj_inject_hook(sv_vec, MID_MULT)
    hook    = (f'blocks.{interv_layer}.hook_resid_pre', hook_fn)
    with model.hooks(fwd_hooks=[hook]):
        with torch.no_grad():
            _, cs = model.run_with_cache(tokens, names_filter=lambda n: hook_name in n, stop_at_layer=PROBE_LAYER+1)
    ste_act = cs[hook_name][0, -1, :].cpu().float().numpy()
    del cs; torch.cuda.empty_cache()
    return sign * (float(ste_act @ w) - float(base_act @ w))


# ── Module A: Layer sweep ─────────────────────────────────────────

SWEEP_CHECKPOINT = f'{DRIVE}/cell5_layer_sweep_raw.json'

if os.path.exists(SWEEP_CHECKPOINT):
    with open(SWEEP_CHECKPOINT) as f:
        raw = json.load(f)
    intervention_results = {em: {int(l): v for l, v in d.items()} for em, d in raw.items()}
    n_done = sum(len(v) for v in intervention_results.values())
    n_total = len(EMOTIONS) * N_LAYERS
    if n_done == n_total:
        print(f'MODULE A: fully complete from checkpoint ({n_done}/{n_total}). Skipping.')
    else:
        print(f'MODULE A: partial checkpoint ({n_done}/{n_total}). Resuming...')
else:
    intervention_results = {em: {} for em in EMOTIONS}
    print(f'MODULE A: Layer sweep  mult={MID_MULT}  n={SAMPLE_N_A}')

for emotion in EMOTIONS:
    sv_vec = sv[emotion]; w = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
    texts  = samples[emotion]
    layers_todo = [l for l in ALL_LAYERS if l not in intervention_results[emotion]]
    if not layers_todo:
        print(f'  {emotion}: all layers done (checkpoint)'); continue
    print(f'\n  {emotion} — {len(layers_todo)} layers remaining...')
    for interv_layer in tqdm(layers_todo, desc=f'  {emotion}'):
        shifts = [_probe_shift(text, w, sign, interv_layer, sv_vec) for text in texts]
        intervention_results[emotion][interv_layer] = float(np.mean(shifts))
        json_safe = {em: {str(l): v for l, v in d.items()} for em, d in intervention_results.items()}
        with open(SWEEP_CHECKPOINT, 'w') as f:
            json.dump(json_safe, f)
    top5 = sorted(intervention_results[emotion].items(), key=lambda x: x[1], reverse=True)[:5]
    print(f'  {emotion} top 5:')
    for layer, shift in top5:
        print(f'    L{layer:2d}: {shift:+.4f}')

with open(SWEEP_CHECKPOINT, 'w') as f:
    json.dump({em: {str(l): v for l, v in d.items()} for em, d in intervention_results.items()}, f, indent=2)
print('\n✅ cell5_layer_sweep_raw.json saved')


# ── Module B: Multiplier sweep ────────────────────────────────────

print('\nMODULE B: Multiplier sweep on top-3 layers per emotion')
CONFIGS = {}
mult_sweep_all = {}

for emotion in EMOTIONS:
    sv_vec = sv[emotion]; w = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
    texts  = samples_b[emotion]
    top3   = [l for l, _ in sorted(intervention_results[emotion].items(), key=lambda x: x[1], reverse=True)[:3]]
    print(f'\n  {emotion} — top-3 layers: {top3}')
    mult_sweep_all[emotion] = {}
    best_both = -1.0; best_layer = top3[0]; best_mult = MULTIPLIERS[0]; best_shift_at_best = -999.0

    for layer in top3:
        mult_sweep_all[emotion][layer] = {}
        for mult in MULTIPLIERS:
            hook_fn = make_proj_inject_hook(sv_vec, mult)
            hook    = (f'blocks.{layer}.hook_resid_pre', hook_fn)
            shifts, cohs = [], []
            for text in texts:
                prompt   = text[:120]
                base_out = generate(prompt, hooks=None, temperature=0.7, max_new_tokens=30)
                base_ps  = _score_text(base_out, w, sign)
                ste_out  = generate(prompt, hooks=[hook], temperature=0.7, max_new_tokens=30)
                ste_ps   = _score_text(ste_out, w, sign)
                shifts.append(ste_ps - base_ps)
                cohs.append(coherence_score(ste_out[len(prompt):].strip()))
            both       = float(np.mean([s > 0 and c >= COH_THRESHOLD for s, c in zip(shifts, cohs)]))
            mean_shift = float(np.mean(shifts))
            mean_coh   = float(np.mean(cohs))
            mult_sweep_all[emotion][layer][mult] = {'mean_shift': mean_shift, 'coherence': mean_coh, 'both': both}
            print(f'    L{layer} ×{mult:4.1f}: shift={mean_shift:+.3f}  coh={mean_coh:.2f}  both={both:.1%}')
            if both > best_both or (both == best_both and mean_shift > best_shift_at_best):
                best_both = both; best_layer = layer; best_mult = mult; best_shift_at_best = mean_shift

    CONFIGS[emotion] = {'layer': best_layer, 'mult': float(best_mult), 'both': float(best_both)}

print('\n' + '='*60)
print('BEST CONFIGS')
for emotion in EMOTIONS:
    b = CONFIGS[emotion]
    print(f'  {emotion:12}: layer={b["layer"]}  mult={b["mult"]}  both={b["both"]:.1%}')

with open(f'{DRIVE}/best_configs.json', 'w') as f:
    json.dump(CONFIGS, f, indent=2)
print('\n✅ best_configs.json saved')

with open(f'{DRIVE}/cell5_mult_sweep_full.json', 'w') as f:
    json.dump({em: {str(l): {str(m): v for m, v in mv.items()} for l, mv in lv.items()} for em, lv in mult_sweep_all.items()}, f, indent=2)
print('✅ cell5_mult_sweep_full.json saved')


# ── Layer heatmap ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, emotion in enumerate(EMOTIONS):
    shifts = intervention_results[emotion]; ax = axes[idx]
    layers_sorted = sorted(shifts.keys()); shift_vals = [shifts[l] for l in layers_sorted]
    best_l = CONFIGS[emotion]['layer']
    bar_colors = ['#C44E52' if l == best_l else '#4C72B0' for l in layers_sorted]
    ax.bar(layers_sorted, shift_vals, color=bar_colors, alpha=0.82)
    ax.axvline(best_l, color='#C44E52', lw=2, ls=':', label=f'Best: L{best_l} ×{CONFIGS[emotion]["mult"]}')
    ax.axvline(PROBE_LAYER, color='gray', lw=1.2, ls='--', alpha=0.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Intervention layer'); ax.set_ylabel('Mean signed probe shift')
    ax.set_title(f'{emotion.capitalize()} — layer sensitivity\nBest: L{best_l} ×{CONFIGS[emotion]["mult"]}  both={CONFIGS[emotion]["both"]:.1%}', fontsize=11, fontweight='bold')
    ax.set_xticks(layers_sorted); ax.set_xticklabels(layers_sorted, fontsize=7)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
fig.suptitle(f'Layer-wise Steering Sensitivity — {MODEL_NAME}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE}/layer_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ layer_heatmap.png saved')

print(f'\n✅ Cell 6 complete')
