#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 9: EMOTIONAL SPACE PCA + AXIS DEFINITION (CPU)
# ═══════════════════════════════════════════════════════════════════
# Extension: Emotional Axes in Llama-3.2-3B
#
# Builds per-tone mean activation vectors, runs PCA, identifies
# the three main emotional axes, and saves everything for Script 10.
#
# Three axes discovered:
#   PC1 (24%): Depth — emotional richness vs detachment/flatness
#   PC2 (19%): Arousal — high energy (frenzy/panic) vs calm
#   PC3 (17%): Dark/Bright — dark narrative vs bright energy
#
# EMNLP three poles in this space:
#   humor:     depth=-4.6  arousal=+0.9  dark=-2.7
#   melancholy:depth=+3.9  arousal=-1.5  dark=+1.7
#   horror:    depth=+0.7  arousal=+2.0  dark=+3.2
#
# No GPU needed — pure numpy/sklearn
# Requires: Script 8 to have completed
#
# Runtime:  ~5 min CPU
# Outputs:  EMOTIONAL_AXES/axis_depth.npy
#           EMOTIONAL_AXES/axis_arousal.npy
#           EMOTIONAL_AXES/axis_darkbright.npy
#           EMOTIONAL_AXES/axis_meta.json
#           EMOTIONAL_AXES/tone_index.json
#           EMOTIONAL_AXES/global_mean.npy
#           EMOTIONAL_AXES/pca_components.npy
#           EMOTIONAL_AXES/emotional_space_final.png
# ═══════════════════════════════════════════════════════════════════

import json, torch
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from google.colab import drive

drive.mount('/content/drive', force_remount=False)

DRIVE_EA = '/content/drive/MyDrive/EMOTIONAL_AXES'

acts_L26 = torch.load(f'{DRIVE_EA}/emotional_acts_L26.pt', map_location='cpu')
df       = pd.read_csv(f'{DRIVE_EA}/emotional_rollouts.csv')
df['response'] = df['response'].fillna('').astype(str)

tones = sorted(df['tone'].unique().tolist())
print(f'Tones: {len(tones)}  Total rows: {len(df)}')

def norm_v(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def get_tone_vec(tone):
    idx = df[(df['tone']==tone) & (df['polarity']=='positive')].index
    return acts_L26[idx].float().numpy().mean(0)

# ── Build tone matrix + center ────────────────────────────────────
X = np.stack([get_tone_vec(t) for t in tones])  # (59, 3072)
global_mean = X.mean(0)
X_centered  = X - global_mean

# ── PCA ───────────────────────────────────────────────────────────
pca    = PCA(n_components=10)
pca.fit(X_centered)
coords = pca.transform(X_centered)  # (59, 10)

print('\nVariance explained by top 10 PCs:')
for i, v in enumerate(pca.explained_variance_ratio_):
    print(f'  PC{i+1}: {v:.4f} ({v*100:.1f}%)')

tone_idx = {t: i for i, t in enumerate(tones)}

# ── Interpret PCs ─────────────────────────────────────────────────
for pc_num in [0, 1, 2]:
    pc = pca.components_[pc_num]
    loadings = [(float(norm_v(X_centered[i]) @ pc), tones[i]) for i in range(len(tones))]
    loadings.sort(reverse=True)
    print(f'\nPC{pc_num+1} top 5 / bottom 5:')
    print(f'  +: {[(t, f"{s:+.3f}") for s,t in loadings[:5]]}')
    print(f'  -: {[(t, f"{s:+.3f}") for s,t in loadings[-5:]]}')

# ── Define axes from PCs (flip signs to intuitive direction) ──────
depth_axis      = pca.components_[0].copy()
arousal_axis    = pca.components_[1].copy()
darkbright_axis = pca.components_[2].copy()

# PC1 positive end = emotional richness (wistfulness, tenderness)
if coords[tone_idx['wistfulness'], 0] < 0: depth_axis = -depth_axis
# PC2 positive end = high arousal (frenzy, panic)
if coords[tone_idx['frenzy'], 1] < 0: arousal_axis = -arousal_axis
# PC3 positive end = dark narrative (horror, grief)
if coords[tone_idx['horror'], 2] < 0: darkbright_axis = -darkbright_axis

# Recompute coords with corrected signs
coords2 = X_centered @ np.stack([depth_axis, arousal_axis, darkbright_axis]).T

# ── Print full summary ────────────────────────────────────────────
print(f'\n{"FINAL EMOTIONAL SPACE":=^65}')
print(f'{"tone":25} {"depth":>8} {"arousal":>8} {"dark/bright":>12}')
print('-'*55)
for tone in sorted(tones):
    i = tone_idx[tone]
    print(f'{tone:25} {coords2[i,0]:>8.2f} {coords2[i,1]:>8.2f} {coords2[i,2]:>12.2f}')

print('\nEMNLP three poles:')
for tone in ['humor','melancholy','horror']:
    i = tone_idx[tone]
    print(f'  {tone:15}: depth={coords2[i,0]:+.3f}  arousal={coords2[i,1]:+.3f}  dark={coords2[i,2]:+.3f}')

# ── Plot: depth vs arousal ────────────────────────────────────────
color_map = {
    'humor':'#2ecc71','joy':'#2ecc71','warmth':'#27ae60','elation':'#2ecc71',
    'excitement':'#27ae60','mischief':'#2ecc71','whimsy':'#2ecc71','hope':'#27ae60',
    'gratitude':'#27ae60','wonder':'#27ae60','awe':'#27ae60','affection':'#27ae60',
    'tenderness':'#27ae60','serenity':'#27ae60','passion':'#e74c3c',
    'melancholy':'#3498db','grief':'#2980b9','sorrow':'#2980b9','despair':'#2c3e50',
    'loneliness':'#3498db','wistfulness':'#3498db','yearning':'#3498db',
    'heartbreak':'#2980b9','numbness':'#7f8c8d','regret':'#3498db',
    'resignation':'#7f8c8d','desolation':'#2c3e50','nostalgia':'#3498db',
    'melancholic_beauty':'#3498db','bittersweet':'#9b59b6','ambivalence':'#9b59b6',
    'horror':'#e74c3c','dread':'#c0392b','anxiety':'#e67e22','paranoia':'#e74c3c',
    'disgust':'#c0392b','unease':'#e67e22','terror':'#c0392b',
    'existential_dread':'#c0392b','foreboding':'#e74c3c','menace':'#c0392b',
    'panic':'#e74c3c','rage':'#c0392b','frenzy':'#c0392b',
    'analytical':'#95a5a6','clinical':'#95a5a6','dry':'#95a5a6',
    'informational':'#95a5a6','sardonic':'#95a5a6','ironic':'#7f8c8d',
    'boredom':'#7f8c8d','flatness':'#7f8c8d','emptiness':'#7f8c8d',
    'detachment':'#95a5a6','exhaustion':'#7f8c8d','apathy':'#7f8c8d',
    'ecstasy':'#f39c12','mania':'#e67e22','reverence':'#9b59b6',
}

fig, axes_plt = plt.subplots(1, 2, figsize=(18, 8))
for ax_idx, (x_col, y_col, x_label, y_label) in enumerate([
    (0, 1, 'Depth (PC1)', 'Arousal (PC2)'),
    (0, 2, 'Depth (PC1)', 'Dark/Bright (PC3)'),
]):
    ax = axes_plt[ax_idx]
    for i, tone in enumerate(tones):
        color = color_map.get(tone, '#2c3e50')
        ax.scatter(coords2[i, x_col], coords2[i, y_col], color=color, s=70, alpha=0.85, zorder=3)
        ax.annotate(tone, (coords2[i, x_col], coords2[i, y_col]),
                    fontsize=6.5, xytext=(3,3), textcoords='offset points', alpha=0.9)
    for tone in ['humor','melancholy','horror']:
        i = tone_idx[tone]
        ax.scatter(coords2[i, x_col], coords2[i, y_col],
                   color='black', s=300, marker='*', zorder=5, label=tone)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4); ax.axvline(0, color='k', lw=0.5, alpha=0.4)
    ax.set_xlabel(x_label, fontsize=10); ax.set_ylabel(y_label, fontsize=10)
    ax.grid(alpha=0.2); ax.legend(fontsize=8)

fig.suptitle('Emotional Space in Llama-3.2-3B  ★ = EMNLP three poles',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE_EA}/emotional_space_final.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ emotional_space_final.png saved')

# ── Save everything ───────────────────────────────────────────────
np.save(f'{DRIVE_EA}/axis_depth.npy',      depth_axis)
np.save(f'{DRIVE_EA}/axis_arousal.npy',    arousal_axis)
np.save(f'{DRIVE_EA}/axis_darkbright.npy', darkbright_axis)
np.save(f'{DRIVE_EA}/global_mean.npy',     global_mean)
np.save(f'{DRIVE_EA}/pca_components.npy',  pca.components_)
np.save(f'{DRIVE_EA}/pca_explained.npy',   pca.explained_variance_ratio_)
np.save(f'{DRIVE_EA}/tone_coords.npy',     coords2)

with open(f'{DRIVE_EA}/tone_index.json','w') as f:
    json.dump(tones, f)

axis_meta = {
    'depth':      {'pc':1, 'variance':float(pca.explained_variance_ratio_[0]),
                   'positive':'emotional richness (wistfulness, tenderness, melancholy)',
                   'negative':'detachment/flatness (clinical, analytical, dry)'},
    'arousal':    {'pc':2, 'variance':float(pca.explained_variance_ratio_[1]),
                   'positive':'high arousal (frenzy, panic, rage, terror)',
                   'negative':'low arousal (serenity, detachment, analytical)'},
    'darkbright': {'pc':3, 'variance':float(pca.explained_variance_ratio_[2]),
                   'positive':'dark narrative (horror, grief, dread, despair)',
                   'negative':'bright energy (elation, joy, excitement, mania)'},
    'emnlp_poles': {
        tone: {'depth':float(coords2[tone_idx[tone],0]),
               'arousal':float(coords2[tone_idx[tone],1]),
               'darkbright':float(coords2[tone_idx[tone],2])}
        for tone in ['humor','melancholy','horror']
    },
    'all_tones': {
        tone: {'depth':float(coords2[tone_idx[tone],0]),
               'arousal':float(coords2[tone_idx[tone],1]),
               'darkbright':float(coords2[tone_idx[tone],2])}
        for tone in tones
    }
}
with open(f'{DRIVE_EA}/axis_meta.json','w') as f:
    json.dump(axis_meta, f, indent=2)

print(f'\n✅ Script 9 complete')
print(f'   Three orthogonal emotional axes saved:')
print(f'     axis_depth.npy      — PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
print(f'     axis_arousal.npy    — PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
print(f'     axis_darkbright.npy — PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% variance)')
