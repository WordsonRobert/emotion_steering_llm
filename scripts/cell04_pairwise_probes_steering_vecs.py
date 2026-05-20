# cell04_pairwise_probes_steering_vecs.py
#
# Trains three pairwise binary probes (humor↔mel, humor↔horror, mel↔horror).
# Builds mean-diff steering vectors in the whitened probe space.
# Verifies geometry and saves all artifacts.
#
# Requires (from cells 1–3):
#   model, DEVICE, D_MODEL, PROBE_LAYER, EMOTIONS, SEEDS
#   df_humor, df_mel, df_hor
#   humor_acts, dark_acts
#   get_activation, normalize_for_probe, norm_v
#
# Produces (in memory + saved to DRIVE):
#   acts_humor, acts_mel, acts_hor   — [N, D_MODEL] float32 numpy
#   w_hm, w_hh, w_mh                — unit probe vectors (numpy float32)
#   mean_hm, std_hm, mean_hh, std_hh
#   sv       — {emotion: tensor on DEVICE bfloat16}
#   PROBE_FOR, PROBE_SIGN
#   Saved: probe_humor_vs_melancholy.pt, probe_humor_vs_horror.pt,
#          probe_melancholy_vs_horror.pt, steering_vecs.pt, cell4_summary.json
#
# RESULT:
#   cos(humor, mel)    = -0.9972
#   cos(humor, horror) = -0.3167
#   cos(mel, horror)   = +0.3151
#
# Runtime: ~25 min on A100

import torch, numpy as np, json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ── Part A: Per-emotion activations ──────────────────────────────

def _extract_df_acts(df, label):
    acts = []
    for text in tqdm(df['text'].tolist(), desc=f'  {label}'):
        try:
            acts.append(get_activation(text).numpy())
        except Exception:
            acts.append(np.zeros(D_MODEL, dtype=np.float32))
    return np.array(acts, dtype=np.float32)


print(f'Part A: Extracting per-emotion activations at PROBE_LAYER={PROBE_LAYER}...')
acts_humor = humor_acts.float().numpy()
print(f'  humor:  {acts_humor.shape}  [reused from cell03 memory]')
acts_mel = _extract_df_acts(df_mel, 'melancholy')
print(f'  mel:    {acts_mel.shape}')
acts_hor = _extract_df_acts(df_hor, 'horror')
print(f'  horror: {acts_hor.shape}')


# ── Part B: Three binary probes ───────────────────────────────────

def _train_probe(acts_pos, acts_neg, label_pos, label_neg, seed):
    n   = min(len(acts_pos), len(acts_neg))
    rng = np.random.default_rng(seed)
    idx_p = rng.choice(len(acts_pos), n, replace=False)
    idx_n = rng.choice(len(acts_neg), n, replace=False)

    X    = np.vstack([acts_pos[idx_p], acts_neg[idx_n]])
    y    = np.array([1]*n + [0]*n)
    mean = X.mean(0); std = X.std(0) + 1e-8
    Xn   = (X - mean) / std
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    Xn   = Xn / norms

    cv  = cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        Xn, y, cv=5, n_jobs=-1
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(Xn, y)
    w   = clf.coef_[0].astype(np.float32)
    w   = w / np.linalg.norm(w)

    print(f'  {label_pos} vs {label_neg}: CV={cv.mean():.4f} ± {cv.std():.4f}  n={n}')
    return w, mean.astype(np.float32), std.astype(np.float32), float(cv.mean()), float(cv.std())


print('\nPart B: Training three binary probes...')
seed = SEEDS['train']

w_hm, mean_hm, std_hm, acc_hm, std_acc_hm = _train_probe(acts_humor, acts_mel,  'humor', 'melancholy', seed)
w_hh, mean_hh, std_hh, acc_hh, std_acc_hh = _train_probe(acts_humor, acts_hor,  'humor', 'horror',     seed)
w_mh, mean_mh, std_mh, acc_mh, std_acc_mh = _train_probe(acts_mel,   acts_hor,  'melancholy', 'horror', seed)

torch.save(torch.tensor(w_hm), f'{DRIVE}/probe_humor_vs_melancholy.pt')
torch.save(torch.tensor(w_hh), f'{DRIVE}/probe_humor_vs_horror.pt')
torch.save(torch.tensor(w_mh), f'{DRIVE}/probe_melancholy_vs_horror.pt')
print('✅ Probe weights saved')

print('\nProbe vector orthogonality (cosines):')
print(f'  cos(humor_mel, humor_horror): {np.dot(w_hm, w_hh):+.4f}')
print(f'  cos(humor_mel, mel_horror)  : {np.dot(w_hm, w_mh):+.4f}')
print(f'  cos(humor_hor, mel_horror)  : {np.dot(w_hh, w_mh):+.4f}')


# ── Part C: Mean-diff steering vectors ───────────────────────────

def _build_steering_vec(acts, mean, std):
    mean_act = acts.mean(0)
    normed   = (mean_act - mean) / (std + 1e-8)
    normed   = normed / np.linalg.norm(normed)
    return normed.astype(np.float32)


print('\nPart C: Building steering vectors...')
steer_humor = _build_steering_vec(acts_humor, mean_hm, std_hm)
steer_mel   = _build_steering_vec(acts_mel,   mean_hm, std_hm)
steer_hor   = _build_steering_vec(acts_hor,   mean_hh, std_hh)

print('Steering vector orthogonality:')
print(f'  humor · melancholy : {np.dot(steer_humor, steer_mel):+.4f}')
print(f'  humor · horror     : {np.dot(steer_humor, steer_hor):+.4f}')
print(f'  melancholy · horror: {np.dot(steer_mel,   steer_hor):+.4f}')

steering_vecs = {
    'humor':      torch.tensor(steer_humor),
    'melancholy': torch.tensor(steer_mel),
    'horror':     torch.tensor(steer_hor),
}
torch.save(steering_vecs, f'{DRIVE}/steering_vecs.pt')
print('✅ steering_vecs.pt saved')

sv = {k: v.to(DEVICE).to(torch.bfloat16) for k, v in steering_vecs.items()}

PROBE_FOR  = {'humor': w_hm, 'melancholy': w_hm, 'horror': w_hh}
PROBE_SIGN = {'humor': +1,   'melancholy': -1,    'horror': -1}


# ── Part D: Geometry verification ────────────────────────────────

print('\nPart D: Geometry verification...')
h_centroid = acts_humor.mean(0)
m_centroid = acts_mel.mean(0)
r_centroid = acts_hor.mean(0)

print(f'  humor centroid → w_hm: {float(h_centroid @ w_hm):+.4f}')
print(f'  mel centroid   → w_hm: {float(m_centroid @ w_hm):+.4f}')
print(f'  horror centroid→ w_hh: {float(r_centroid @ w_hh):+.4f}')


# ── Save summary ──────────────────────────────────────────────────

cell4_summary = {
    'probe_accs': {
        'humor_vs_melancholy':  {'cv_mean': acc_hm, 'cv_std': std_acc_hm},
        'humor_vs_horror':      {'cv_mean': acc_hh, 'cv_std': std_acc_hh},
        'melancholy_vs_horror': {'cv_mean': acc_mh, 'cv_std': std_acc_mh},
    },
    'probe_orthogonality': {
        'hm_dot_hh': float(np.dot(w_hm, w_hh)),
        'hm_dot_mh': float(np.dot(w_hm, w_mh)),
        'hh_dot_mh': float(np.dot(w_hh, w_mh)),
    },
    'steering_vec_orthogonality': {
        'humor_dot_mel':    float(np.dot(steer_humor, steer_mel)),
        'humor_dot_horror': float(np.dot(steer_humor, steer_hor)),
        'mel_dot_horror':   float(np.dot(steer_mel,   steer_hor)),
    },
    'PROBE_LAYER': PROBE_LAYER,
}
with open(f'{DRIVE}/cell4_summary.json', 'w') as f:
    json.dump(cell4_summary, f, indent=2)

print(f'\n━━━ CELL 4 SUMMARY ━━━')
print(f'  probe_humor_vs_melancholy  CV={acc_hm:.4f} ± {std_acc_hm:.4f}')
print(f'  probe_humor_vs_horror      CV={acc_hh:.4f} ± {std_acc_hh:.4f}')
print(f'  probe_melancholy_vs_horror CV={acc_mh:.4f} ± {std_acc_mh:.4f}')
print(f'\n  humor · mel     = {np.dot(steer_humor, steer_mel):+.4f}')
print(f'  humor · horror  = {np.dot(steer_humor, steer_hor):+.4f}')
print(f'  mel · horror    = {np.dot(steer_mel, steer_hor):+.4f}')
print(f'\n✅ Cell 4 complete')
