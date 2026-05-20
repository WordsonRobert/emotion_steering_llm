# cell14_tasks_3_4_5.py
#
# CPU-only analyses — no model forward passes needed.
#
# Task 3: Binary vs Three-Pole Geometry
#   Off-diagonal bleed asymmetry: mel↔horror 7.5% vs humor↔other 3.8% (ratio 2.0x)
#   Supports circumplex geometry: mel/horror share a valence axis.
#
# Task 4: Surface Feature Probe
#   10 handcrafted features (TTR, punct density, sent count, etc.)
#   Surface probe CV acc = 0.788 ± 0.012
#   Activation probe acc = 0.900
#   Gap = +0.112 (activation captures 11.2% more)
#
# Task 5: OOD Evaluation
#   Humor OOD (english_quotes):  87.6% → 59.0%  drop −28.6%
#   Mel OOD (merve/poetry):      96.9% → 31.0%  drop −65.9%
#   Horror OOD (imdb negative):  22.6% → 28.5%  drop −5.9%
#
# Requires on disk: simplex_results_T07.csv, steering_vecs.pt, master CSVs
# Requires packages: datasets (HuggingFace)
#
# Runtime: ~25 min (mostly HuggingFace dataset downloads)

import numpy as np, pandas as pd, json, os, re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

DRIVE    = '/content/drive/MyDrive/STEERING_EMNLP_2026'
EMOTIONS = ['humor', 'melancholy', 'horror']

df_results = pd.read_csv(f'{DRIVE}/simplex_results_T07.csv')
df_humor   = pd.read_csv(f'{DRIVE}/humor_master.csv')
df_mel     = pd.read_csv(f'{DRIVE}/melancholy_master.csv')
df_hor     = pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor, df_mel, df_hor]:
    df['text'] = df['text'].fillna('').astype(str).str.strip()

try:
    sv_raw = torch.load(f'{DRIVE}/steering_vecs.pt', map_location='cpu')
    sv_np  = {k: v.float().numpy() for k, v in sv_raw.items()}
    print('✅ steering_vecs loaded')
except Exception as e:
    sv_np = None; print(f'⚠ steering_vecs not available: {e}')

print(f'df_results: {len(df_results)} rows')


# ── Task 3: Binary vs Three-Pole Geometry ─────────────────────────

print(f'\n{"="*60}')
print('TASK 3: Binary vs Three-Pole Geometry')

conf = np.zeros((3, 3))
for i, te in enumerate(EMOTIONS):
    sub = df_results[df_results['emotion_target'] == te]
    for j, pe in enumerate(EMOTIONS):
        conf[i, j] = (sub['pred_emotion'] == pe).sum() / max(len(sub), 1)
print(f'  {"":12} {"→humor":>10} {"→mel":>10} {"→horror":>10}')
for i, em in enumerate(EMOTIONS):
    print(f'  {em:12} {conf[i,0]:>10.1%} {conf[i,1]:>10.1%} {conf[i,2]:>10.1%}')

mel_horror_bleed  = (conf[1,2] + conf[2,1]) / 2
humor_other_bleed = (conf[0,1] + conf[0,2] + conf[1,0] + conf[2,0]) / 4
print(f'\n  mel↔horror bleed (avg): {mel_horror_bleed:.1%}')
print(f'  humor↔other bleed (avg): {humor_other_bleed:.1%}')
print(f'  Ratio: {mel_horror_bleed/max(humor_other_bleed,1e-8):.2f}x')

if sv_np:
    print('\n  Steering vector cosine matrix:')
    for i, a in enumerate(EMOTIONS):
        for j, b in enumerate(EMOTIONS):
            if j > i:
                va = sv_np[a] / (np.linalg.norm(sv_np[a]) + 1e-8)
                vb = sv_np[b] / (np.linalg.norm(sv_np[b]) + 1e-8)
                print(f'  cos({a}, {b}) = {float(va @ vb):+.4f}')

task3_results = {
    'mel_horror_bleed': float(mel_horror_bleed),
    'humor_other_bleed': float(humor_other_bleed),
    'bleed_ratio': float(mel_horror_bleed / max(humor_other_bleed, 1e-8)),
    'confusion_matrix': conf.tolist(),
}
with open(f'{DRIVE}/task3_binary_vs_threepole_results.json', 'w') as f:
    json.dump(task3_results, f, indent=2)
print('✅ task3_binary_vs_threepole_results.json saved')


# ── Task 4: Surface Feature Probe ─────────────────────────────────

print(f'\n{"="*60}')
print('TASK 4: Surface Feature Probe')

def extract_surface_features(texts):
    feats = []
    for text in texts:
        words = text.split(); sents = re.split(r'[.!?]+', text)
        sents = [s.strip() for s in sents if s.strip()]; chars = list(text)
        feats.append({
            'ttr':           len(set(words)) / max(len(words), 1),
            'punct_density': sum(1 for c in chars if c in '.,!?;:') / max(len(chars), 1),
            'sent_count':    len(sents),
            'avg_word_len':  np.mean([len(w) for w in words]) if words else 0,
            'exclaim_rate':  text.count('!') / max(len(words), 1),
            'question_rate': text.count('?') / max(len(words), 1),
            'upper_rate':    sum(1 for c in chars if c.isupper()) / max(len(chars), 1),
            'avg_sent_len':  np.mean([len(s.split()) for s in sents]) if sents else 0,
            'ellipsis_rate': text.count('...') / max(len(words), 1),
            'digit_rate':    sum(1 for c in chars if c.isdigit()) / max(len(chars), 1),
        })
    return pd.DataFrame(feats)

all_texts  = df_humor['text'].tolist() + df_mel['text'].tolist() + df_hor['text'].tolist()
all_labels = [0]*len(df_humor) + [1]*len(df_mel) + [2]*len(df_hor)

print(f'  Extracting surface features for {len(all_texts)} texts...')
X_surf = extract_surface_features(all_texts).values
y      = np.array(all_labels)
scaler = StandardScaler(); X_surf = scaler.fit_transform(X_surf)

clf_surf = LogisticRegression(max_iter=1000, C=1.0)
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
surf_cv  = cross_val_score(clf_surf, X_surf, y, cv=skf)
surf_acc = float(surf_cv.mean()); surf_std = float(surf_cv.std())
act_acc  = float(df_results['pred_correct'].mean())
gap      = act_acc - surf_acc

print(f'\n  Surface probe  CV acc: {surf_acc:.3f} ± {surf_std:.3f}')
print(f'  Activation probe acc: {act_acc:.3f}')
print(f'  Gap: {gap:+.3f}  (activation captures {gap:.1%} more than surface features)')

clf_surf.fit(X_surf, y)
feat_names   = ['ttr','punct_density','sent_count','avg_word_len','exclaim_rate',
                'question_rate','upper_rate','avg_sent_len','ellipsis_rate','digit_rate']
importances  = np.abs(clf_surf.coef_).mean(axis=0)
ranked_feats = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
print('\n  Feature importances (mean |coef|):')
for fname, imp in ranked_feats:
    print(f'    {fname:20s}: {imp:.4f}')

task4_results = {'surface_probe_acc': surf_acc, 'surface_probe_std': surf_std,
                 'activation_probe_acc': act_acc, 'gap': gap,
                 'feature_importances': {f: float(i) for f, i in ranked_feats}}
with open(f'{DRIVE}/task4_surface_probe_results.json', 'w') as f:
    json.dump(task4_results, f, indent=2)
print('✅ task4_surface_probe_results.json saved')


# ── Task 5: OOD Evaluation ────────────────────────────────────────

print(f'\n{"="*60}')
print('TASK 5: OOD Evaluation')

from datasets import load_dataset
clf_ood = LogisticRegression(max_iter=1000, C=1.0)
clf_ood.fit(X_surf, y)
ood_results = {}

# Humor OOD: english_quotes
print('\n  Loading humor OOD (english_quotes)...')
try:
    ds_quotes   = load_dataset('Abirate/english_quotes', split='train')
    quotes_texts = [r['quote'] for r in ds_quotes if isinstance(r['quote'], str) and 30 < len(r['quote']) < 200][:200]
    X_ood_h     = scaler.transform(extract_surface_features(quotes_texts).values)
    preds_h     = clf_ood.predict(X_ood_h)
    acc_h       = float((preds_h == 0).mean())
    ood_results['humor_ood'] = {'acc': acc_h, 'n': len(quotes_texts), 'source': 'english_quotes'}
    print(f'  Humor OOD acc: {acc_h:.3f}  (n={len(quotes_texts)})')
except Exception as e:
    print(f'  Humor OOD failed: {e}')
    ood_results['humor_ood'] = {'acc': None, 'n': 0, 'source': 'english_quotes'}

# Melancholy OOD: poetry
print('\n  Loading melancholy OOD (merve/poetry)...')
try:
    ds_poetry    = load_dataset('merve/poetry', split='train')
    sad_keywords = ['sad','sorrow','grief','loss','pain','lonely','dark','tears','death','despair','melancholy','weep']
    mel_texts    = []
    for r in ds_poetry:
        content = r.get('content', '') or r.get('poem', '') or ''
        if any(kw in content.lower() for kw in sad_keywords) and 30 < len(content) < 300:
            mel_texts.append(content[:200])
        if len(mel_texts) >= 200: break
    if len(mel_texts) < 20:
        mel_texts = [str(r.get('content', '') or r.get('poem', ''))[:200] for r in ds_poetry if len(str(r.get('content',''))) > 30][:200]
    X_ood_m = scaler.transform(extract_surface_features(mel_texts).values)
    acc_m   = float((clf_ood.predict(X_ood_m) == 1).mean())
    ood_results['mel_ood'] = {'acc': acc_m, 'n': len(mel_texts), 'source': 'merve/poetry'}
    print(f'  Mel OOD acc: {acc_m:.3f}  (n={len(mel_texts)})')
except Exception as e:
    print(f'  Mel OOD failed: {e}')
    ood_results['mel_ood'] = {'acc': None, 'n': 0, 'source': 'merve/poetry'}

# Horror OOD: IMDB negative
print('\n  Loading horror OOD (imdb negative)...')
try:
    ds_imdb      = load_dataset('stanfordnlp/imdb', split='test')
    horror_texts = [r['text'][:200] for r in ds_imdb if r['label'] == 0 and len(r['text']) > 50][:200]
    X_ood_r      = scaler.transform(extract_surface_features(horror_texts).values)
    acc_r        = float((clf_ood.predict(X_ood_r) == 2).mean())
    ood_results['horror_ood'] = {'acc': acc_r, 'n': len(horror_texts), 'source': 'imdb_neg'}
    print(f'  Horror OOD acc: {acc_r:.3f}  (n={len(horror_texts)})')
except Exception as e:
    print(f'  Horror OOD failed: {e}')
    ood_results['horror_ood'] = {'acc': None, 'n': 0, 'source': 'imdb_neg'}

print(f'\n{"="*60}')
print('TASK 5 SUMMARY')
for i, em in enumerate(EMOTIONS):
    mask    = y == i
    id_acc  = float((clf_ood.predict(X_surf[mask]) == i).mean())
    ood_key = f'{em if em != "melancholy" else "mel"}_ood'
    ood_acc = ood_results.get(ood_key, {}).get('acc', None)
    if ood_acc is not None:
        print(f'  {em:12}: in-dist={id_acc:.3f}  OOD={ood_acc:.3f}  drop={id_acc-ood_acc:+.3f}')
    else:
        print(f'  {em:12}: in-dist={id_acc:.3f}  OOD=N/A')

with open(f'{DRIVE}/task5_ood_results.json', 'w') as f:
    json.dump({'in_dist_surface_acc': surf_acc, 'ood_results': ood_results}, f, indent=2)
print('✅ task5_ood_results.json saved')

print(f'\n━━━ CELL 14 COMPLETE ━━━')
