# cell03_activation_extraction_probe.py
#
# Part A: Extract last-token residual activations at PROBE_LAYER for all texts.
# Part B: Train a binary probe (humor vs dark) at every layer (0–27) via 5-fold CV.
# Part C: Plot probe accuracy by layer.
#
# Requires (from cells 1–2):
#   model, N_LAYERS, D_MODEL, PROBE_LAYER, ALL_LAYERS
#   SEEDS, df_humor, dark_texts, N_HUMOR, N_DARK
#   get_activation, get_activation_at_layer, normalize_for_probe
#
# Produces:
#   humor_acts        — [N_HUMOR, D_MODEL] float32 tensor (in memory)
#   dark_acts         — [N_DARK,  D_MODEL] float32 tensor (in memory)
#   probe_weights     — dict {layer: tensor([D_MODEL])}
#   layer_probe_accs  — dict {layer: float}
#   Saved to DRIVE: humor_acts.pt, dark_acts.pt, probe_weights.pt
#                   layer_probe_accuracy.png
#
# RESULT: L26 CV accuracy = 0.9417 ± 0.0091
#
# Runtime: ~45 min on A100

# ── Part A: Extract activations at PROBE_LAYER ────────────────────

def _extract_acts(texts, label):
    acts = []
    for text in tqdm(texts, desc=f'  {label}'):
        try:
            acts.append(get_activation(text))
        except Exception:
            acts.append(torch.zeros(D_MODEL))
    return torch.stack(acts)


print(f'Part A: Extracting activations at PROBE_LAYER={PROBE_LAYER}...')
humor_acts = _extract_acts(df_humor['text'].tolist(), 'humor')
dark_acts  = _extract_acts(dark_texts, 'dark')

torch.save(humor_acts, f'{DRIVE}/humor_acts.pt')
torch.save(dark_acts,  f'{DRIVE}/dark_acts.pt')
print(f'  ✅ humor_acts: {tuple(humor_acts.shape)}')
print(f'  ✅ dark_acts:  {tuple(dark_acts.shape)}')


# ── Part B: Train binary probe at every layer ─────────────────────

N_PROBE = min(300, N_HUMOR, N_DARK)
print(f'\nPart B: Training binary probe at every layer...')
print(f'  N_PROBE={N_PROBE}  |  Layers={N_LAYERS}  |  ~40 min on A100')

probe_weights    = {}
layer_probe_accs = {}

rng = np.random.default_rng(SEEDS['train'])
h_idx_probe = rng.choice(N_HUMOR, N_PROBE, replace=False)
d_idx_probe = rng.choice(N_DARK,  N_PROBE, replace=False)

for layer in tqdm(ALL_LAYERS, desc='  probe per layer'):
    h_acts_l, d_acts_l = [], []
    for i in h_idx_probe:
        try:
            h_acts_l.append(get_activation_at_layer(df_humor['text'].iloc[i], layer).numpy())
        except Exception:
            h_acts_l.append(np.zeros(D_MODEL))
    for i in d_idx_probe:
        try:
            d_acts_l.append(get_activation_at_layer(dark_texts[i], layer).numpy())
        except Exception:
            d_acts_l.append(np.zeros(D_MODEL))

    X = np.vstack([h_acts_l, d_acts_l])
    y = np.array([1]*N_PROBE + [0]*N_PROBE)

    Xn, _, _ = normalize_for_probe(X)

    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0, random_state=SEEDS['train']),
        Xn, y, cv=5, n_jobs=-1
    )
    layer_probe_accs[layer] = float(cv_scores.mean())

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEEDS['train'])
    clf.fit(Xn, y)
    w = clf.coef_[0].astype(np.float32)
    w = w / np.linalg.norm(w)
    probe_weights[layer] = torch.tensor(w)

torch.save(probe_weights, f'{DRIVE}/probe_weights.pt')
print(f'  ✅ probe_weights.pt saved ({len(probe_weights)} layers)')

X_check = np.vstack([
    humor_acts[h_idx_probe].numpy(),
    dark_acts[d_idx_probe].numpy(),
])
y_check = np.array([1]*N_PROBE + [0]*N_PROBE)
Xn_check, _, _ = normalize_for_probe(X_check)
cv_check = cross_val_score(
    LogisticRegression(max_iter=1000, C=1.0),
    Xn_check, y_check, cv=5
)
print(f'\n  Probe L{PROBE_LAYER} CV accuracy (from stored acts): '
      f'{cv_check.mean():.4f} ± {cv_check.std():.4f}')


# ── Part C: Layer accuracy plot ───────────────────────────────────

layers_sorted = sorted(layer_probe_accs.keys())
accs_sorted   = [layer_probe_accs[l] for l in layers_sorted]
best_layer    = max(layer_probe_accs, key=layer_probe_accs.get)

print('\n  Accuracy per layer:')
for l in layers_sorted:
    marker = ' ← PROBE_LAYER' if l == PROBE_LAYER else (
             ' ← BEST'        if l == best_layer  else '')
    print(f'    L{l:2d}: {layer_probe_accs[l]:.4f}{marker}')

fig, ax = plt.subplots(figsize=(13, 4))
bar_colors = [
    '#C44E52' if l == PROBE_LAYER else
    '#DD8452' if l == best_layer  else
    '#4C72B0'
    for l in layers_sorted
]
ax.bar(layers_sorted, accs_sorted, color=bar_colors, alpha=0.85)
ax.axhline(0.5, color='k', lw=1, ls='--', alpha=0.5, label='Chance')
ax.axvline(PROBE_LAYER, color='#C44E52', lw=2, ls=':',
           label=f'PROBE_LAYER={PROBE_LAYER} (acc={layer_probe_accs[PROBE_LAYER]:.3f})')
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('CV Accuracy (5-fold)', fontsize=11)
ax.set_title(
    f'Binary Probe Accuracy by Layer\n'
    f'humor vs dark (mel+horror) | {MODEL_NAME} | N={N_PROBE} per class',
    fontsize=12, fontweight='bold'
)
ax.set_xticks(layers_sorted)
ax.set_xticklabels(layers_sorted, fontsize=8)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DRIVE}/layer_probe_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'  ✅ layer_probe_accuracy.png saved')

print(f'\n━━━ CELL 3 SUMMARY ━━━')
print(f'  humor_acts : {tuple(humor_acts.shape)}')
print(f'  dark_acts  : {tuple(dark_acts.shape)}')
print(f'  PROBE_LAYER acc: {layer_probe_accs[PROBE_LAYER]:.4f}')
print(f'  Best layer acc : L{best_layer} = {layer_probe_accs[best_layer]:.4f}')
print(f'\n✅ Cell 3 complete')
