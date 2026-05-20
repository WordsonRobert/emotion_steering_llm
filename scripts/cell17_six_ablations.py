# cell17_six_ablations.py
#
# Six ablation experiments. max_new_tokens=20, ABL_N=15.
#
# Exp A: Random vector baseline — our vector vs 3 random vectors (3 seeds).
#   RESULT: humor our=40% vs rand=48.9% p=0.008** (n=15 artifact);
#           horror our=86.7% vs rand=46.7% p=0.0002***
#
# Exp B: Data size ablation on humor — sizes 25 to 500.
#   RESULT: no monotonic trend at n=15; both ranges 26.7%–60.0%
#
# Exp C: RepE PCA vs Mean-Diff vector comparison.
#   RESULT: cos(PCA, mean_diff)=−0.2033 (nearly orthogonal)
#           cos(MD, probe_w)=+0.8848 (mean-diff aligns with probe)
#
# Exp D: Head ablation control — subject-verb agreement accuracy
#   when critical heads are zeroed.
#   RESULT: SV baseline acc=0.300; mean specificity=+0.020
#
# Exp E: 4-Turn Tonal Drift (horror, n=15).
#   RESULT: T1=+6.892 → T2=+4.834; T1 vs T2 ns (p≈0.08)
#
# Exp F: DistilRoBERTa independent evaluation on steered outputs.
#   NOTE: This cell runs Exp F with a broken column reference.
#         See cell19 for the corrected version (uses 'ste_gen' column).
#
# Requires (from cell15): model, device, EMOTIONS, PROBE_LAYER, DRIVE, CONFIGS
#   df_humor, df_mel, df_hor, sv, PROBE_FOR, PROBE_SIGN
#   coherence_score, probe_score, build_hook, humor_acts.pt, dark_acts.pt
#   cell9_summary.json (critical_heads from cell13)
#
# Runtime: ~1.5 hrs on A100

import torch, numpy as np, pandas as pd, json, os, math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from collections import Counter
from tqdm import tqdm

DRIVE     = '/content/drive/MyDrive/STEERING_EMNLP_2026'
EVAL_TEMP = 0.7
actual_t  = 1e-10 if EVAL_TEMP == 0.0 else EVAL_TEMP
ABL_N     = 15

with open(f'{DRIVE}/cell9_summary.json') as f:
    cell9 = json.load(f)
critical_heads = cell9['critical_heads']

humor_acts_pt = torch.load(f'{DRIVE}/humor_acts.pt',    map_location='cpu').float()
dark_acts_pt  = torch.load(f'{DRIVE}/dark_acts.pt',     map_location='cpu').float()
probe_weights = torch.load(f'{DRIVE}/probe_weights.pt', map_location='cpu')

print(f'CELL 17: Six Ablation Experiments')
print(f'  EVAL_TEMP={EVAL_TEMP}  ABL_N={ABL_N}  max_new_tokens=20')
print(f'  Critical heads: {len(critical_heads)}')

results_all = {}


# ── Shared helpers ────────────────────────────────────────────────

def run_steering_eval(emotion, hooks, prompts, label):
    w = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]; rows = []
    for prompt in prompts:
        p = prompt[:120]
        base_out = model.generate(p, max_new_tokens=20, temperature=actual_t, verbose=False)
        base_ps  = probe_score(base_out, w, sign)
        with model.hooks(fwd_hooks=hooks):
            ste_out = model.generate(p, max_new_tokens=20, temperature=actual_t, verbose=False)
        ste_ps = probe_score(ste_out, w, sign)
        coh    = coherence_score(ste_out[len(p):].strip())
        rows.append({'emotion': emotion, 'label': label,
                     'ps_shift': float(ste_ps - base_ps),
                     'both': bool((ste_ps > base_ps) and (coh >= 0.6)),
                     'pred_correct': bool(ste_ps > 0), 'coherent': bool(coh >= 0.6)})
    return rows

def norm_v_np(v): n = np.linalg.norm(v); return v / n if n > 1e-8 else v

def make_hook_from_np(vec_np, emotion):
    cfg = CONFIGS[emotion]; layer = int(cfg['layer']); mult = float(cfg['mult'])
    vec = torch.tensor(vec_np, dtype=torch.bfloat16).to(device)
    vec = vec / vec.norm().clamp(min=1e-8)
    def fn(resid_pre, hook):
        proj = (resid_pre @ vec).unsqueeze(-1) * vec
        return (resid_pre - proj) + vec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]


# ── Exp A: Random Vector Baseline ────────────────────────────────

print(f'\n{"="*60}\nEXP A: Random Vector Baseline')
exp_a_rows = []
for emotion in EMOTIONS:
    prompts = pd.concat([df_humor, df_mel, df_hor]).sample(ABL_N, random_state=11)['text'].tolist()
    print(f'  {emotion} our_vec ...', end='', flush=True)
    rows_ours = run_steering_eval(emotion, build_hook(emotion), prompts, 'our_vec')
    print(f'  both={np.mean([r["both"] for r in rows_ours]):.1%}')
    exp_a_rows.extend(rows_ours)
    for seed in [101, 202, 303]:
        rng   = np.random.default_rng(seed)
        r_vec = norm_v_np(rng.standard_normal(D_MODEL).astype(np.float32))
        exp_a_rows.extend(run_steering_eval(emotion, make_hook_from_np(r_vec, emotion), prompts, f'random_{seed}'))

df_a = pd.DataFrame(exp_a_rows)
df_a.to_csv(f'{DRIVE}/task_random_baseline.csv', index=False)
print(f'\n  Summary:')
for emotion in EMOTIONS:
    sub = df_a[df_a['emotion'] == emotion]
    ours = sub[sub['label']=='our_vec']['both'].mean()
    rand = sub[sub['label'].str.startswith('random')]['both'].mean()
    u, p = scipy_stats.mannwhitneyu(sub[sub['label']=='our_vec']['ps_shift'].values,
                                    sub[sub['label'].str.startswith('random')]['ps_shift'].values,
                                    alternative='greater')
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    print(f'  {emotion:12}: our={ours:.1%}  rand={rand:.1%}  p={p:.4f} {sig}')
results_all['exp_a'] = df_a.groupby(['emotion','label'])['both'].mean().to_dict()


# ── Exp B: Data Size Ablation ─────────────────────────────────────

print(f'\n{"="*60}\nEXP B: Data Size Ablation')
emotion   = 'humor'
prompts_b = df_humor.sample(ABL_N, random_state=22).reset_index(drop=True)['text'].tolist()
min_size  = min(len(humor_acts_pt), len(dark_acts_pt))
sizes     = [s for s in [25, 50, 100, 150, 200, 300, min_size] if s <= min_size]
exp_b_rows = []
for size in tqdm(sizes, desc='  Data sizes'):
    rng   = np.random.default_rng(42)
    h_idx = rng.choice(len(humor_acts_pt), size=size, replace=False)
    d_idx = rng.choice(len(dark_acts_pt),  size=size, replace=False)
    md_vec = norm_v_np(humor_acts_pt[h_idx].numpy().mean(0) - dark_acts_pt[d_idx].numpy().mean(0)).astype(np.float32)
    rows = run_steering_eval(emotion, make_hook_from_np(md_vec, emotion), prompts_b, f'size_{size}')
    both = np.mean([r['both'] for r in rows])
    for r in rows: r['data_size'] = size
    exp_b_rows.extend(rows)
    print(f'  size={size:4d}: both={both:.1%}')

df_b = pd.DataFrame(exp_b_rows)
df_b.to_csv(f'{DRIVE}/task_data_ablation.csv', index=False)
results_all['exp_b'] = df_b.groupby('data_size')['both'].mean().to_dict()


# ── Exp C: RepE PCA vs Mean-Diff ──────────────────────────────────

print(f'\n{"="*60}\nEXP C: RepE PCA vs Mean-Diff')
prompts_c = {em: df.sample(ABL_N, random_state=33).reset_index(drop=True)['text'].tolist()
             for em, df in [('humor', df_humor), ('melancholy', df_mel), ('horror', df_hor)]}
stacked   = np.vstack([humor_acts_pt.numpy(), dark_acts_pt.numpy()])
pca       = PCA(n_components=1); pca.fit(stacked - stacked.mean(0))
pca_vec   = pca.components_[0].astype(np.float32)
md_vec    = norm_v_np(humor_acts_pt.numpy().mean(0) - dark_acts_pt.numpy().mean(0)).astype(np.float32)
w_hm_norm = norm_v_np(w_hm.astype(np.float32))

print(f'  cos(PCA, mean_diff) = {float(norm_v_np(pca_vec) @ norm_v_np(md_vec)):+.4f}')
print(f'  cos(PCA, probe_w)   = {float(norm_v_np(pca_vec) @ w_hm_norm):+.4f}')
print(f'  cos(MD,  probe_w)   = {float(norm_v_np(md_vec)  @ w_hm_norm):+.4f}')

exp_c_rows = []
for emotion in EMOTIONS:
    for vec_name, vec in [('pca', pca_vec), ('mean_diff', md_vec)]:
        print(f'  {emotion} {vec_name} ...', end='', flush=True)
        rows = run_steering_eval(emotion, make_hook_from_np(vec, emotion), prompts_c[emotion], vec_name)
        print(f'  both={np.mean([r["both"] for r in rows]):.1%}')
        exp_c_rows.extend(rows)

df_c = pd.DataFrame(exp_c_rows)
df_c.to_csv(f'{DRIVE}/task_repe_comparison.csv', index=False)
results_all['exp_c'] = {
    'cos_pca_md':    float(norm_v_np(pca_vec) @ norm_v_np(md_vec)),
    'cos_pca_probe': float(norm_v_np(pca_vec) @ w_hm_norm),
    'cos_md_probe':  float(norm_v_np(md_vec)  @ w_hm_norm),
}
print(f'\n  Summary:')
for emotion in EMOTIONS:
    sub = df_c[df_c['emotion'] == emotion]
    print(f'  {emotion:12}: PCA={sub[sub["label"]=="pca"]["both"].mean():.1%}  '
          f'mean_diff={sub[sub["label"]=="mean_diff"]["both"].mean():.1%}')


# ── Exp D: Head Control (Subject-Verb Agreement) ──────────────────

print(f'\n{"="*60}\nEXP D: Head Ablation Control — Subject-Verb Agreement')
sv_pairs = [
    ("The keys to the cabinet",    "are",    "is"),
    ("The dog near the trees",     "runs",   "run"),
    ("The manager of the stores",  "was",    "were"),
    ("The books on the shelf",     "are",    "is"),
    ("The girl with the cats",     "plays",  "play"),
    ("The author of the novels",   "writes", "write"),
    ("The students in the class",  "study",  "studies"),
    ("The bird by the windows",    "sings",  "sing"),
    ("The child near the boxes",   "cries",  "cry"),
    ("The teacher with the markers","draws", "draw"),
    ("The cat on the chairs",      "sleeps", "sleep"),
    ("The player near the goals",  "scores", "score"),
    ("The woman with the bags",    "walks",  "walk"),
    ("The man by the rivers",      "fishes", "fish"),
    ("The nurse at the clinics",   "works",  "work"),
    ("The pilot near the planes",  "flies",  "fly"),
    ("The chef in the kitchens",   "cooks",  "cook"),
    ("The artist near the walls",  "paints", "paint"),
    ("The driver by the trucks",   "parks",  "park"),
    ("The farmer with the cows",   "milks",  "milk"),
] * 2  # 40 pairs

def sv_accuracy(ablate_layer=None, ablate_head=None):
    correct = 0
    for subj, gram, ungram in sv_pairs:
        ctx = model.to_tokens([subj], prepend_bos=True)
        hooks = []
        if ablate_layer is not None:
            h = ablate_head
            def make_z(hi):
                def fn(z, hook):
                    z_new = z.clone(); z_new[:,:,hi,:] = 0.0; return z_new
                return fn
            hooks = [(f'blocks.{ablate_layer}.attn.hook_z', make_z(h))]
        with torch.no_grad():
            if hooks:
                with model.hooks(fwd_hooks=hooks): logits = model(ctx)
            else: logits = model(ctx)
        last  = logits[0, -1, :]
        g_id  = model.to_tokens([gram],   prepend_bos=False)[0, 0]
        u_id  = model.to_tokens([ungram], prepend_bos=False)[0, 0]
        if last[g_id] > last[u_id]: correct += 1
    return correct / len(sv_pairs)

print('  Baseline SV accuracy ...', end='', flush=True)
sv_base = sv_accuracy(); print(f' {sv_base:.3f}')

exp_d_rows = []
for ch in tqdm(critical_heads[:10], desc='  Critical heads'):
    l, h     = int(ch['layer']), int(ch['head'])
    emo_drop = float(ch['drop'])
    sv_abl   = sv_accuracy(ablate_layer=l, ablate_head=h)
    sv_drop  = sv_base - sv_abl
    spec     = emo_drop - sv_drop
    exp_d_rows.append({'layer': l, 'head': h, 'emo_drop': emo_drop,
                       'sv_drop': sv_drop, 'sv_ablated_acc': sv_abl, 'specificity': spec})
    print(f'  L{l:2d}H{h:2d}: emo_drop={emo_drop:.3f}  sv_drop={sv_drop:.3f}  spec={spec:+.3f}')

df_d = pd.DataFrame(exp_d_rows)
df_d.to_csv(f'{DRIVE}/task_head_ablation_control.csv', index=False)
results_all['exp_d'] = {'sv_baseline': sv_base, 'mean_specificity': float(df_d['specificity'].mean())}
print(f'  Mean specificity: {df_d["specificity"].mean():+.3f}')


# ── Exp E: 4-Turn Tonal Drift ─────────────────────────────────────

print(f'\n{"="*60}\nEXP E: 4-Turn Tonal Drift')
DRIFT4_N       = 15
drift4_prompts = df_hor.sample(DRIFT4_N, random_state=66).reset_index(drop=True)['text'].tolist()
neutral_turns  = ["What happened next?", "Tell me more.", "And then?", "Continue."] * 10
horror_hooks   = build_hook('horror')

def horror_score(text): return probe_score(text, PROBE_FOR['horror'], PROBE_SIGN['horror'])

exp_e_rows = []
for i, prompt in enumerate(tqdm(drift4_prompts, desc='  4-turn pairs')):
    p = prompt[:120]
    with model.hooks(fwd_hooks=horror_hooks):
        t1 = model.generate(p, max_new_tokens=40, temperature=actual_t, verbose=False)
    scores = [horror_score(t1)]; ctx = t1
    for turn in range(2, 5):
        nu  = neutral_turns[(i*3+turn) % len(neutral_turns)]
        ctx = (ctx + ' ' + nu)[-400:]
        out = model.generate(ctx, max_new_tokens=40, temperature=actual_t, verbose=False)
        scores.append(horror_score(out)); ctx = out
    exp_e_rows.append({'pair_id': i,
                       'score_t1': scores[0], 'score_t2': scores[1],
                       'score_t3': scores[2], 'score_t4': scores[3],
                       'drift_t2': scores[0]-scores[1]})

df_e = pd.DataFrame(exp_e_rows)
df_e.to_csv(f'{DRIVE}/task_4turn_drift.csv', index=False)
print(f'\n  Turn scores (mean):')
for t in [1,2,3,4]: print(f'  Turn {t}: {df_e[f"score_t{t}"].mean():+.4f}')
for t in [2,3,4]:
    u, p = scipy_stats.mannwhitneyu(df_e['score_t1'].values, df_e[f'score_t{t}'].values, alternative='two-sided')
    sig  = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    print(f'  T1 vs T{t}: U={u:.0f}  p={p:.4f}  {sig}')
results_all['exp_e'] = {f'mean_t{t}': float(df_e[f'score_t{t}'].mean()) for t in [1,2,3,4]}


# ── Exp F: DistilRoBERTa (broken — see cell19 for fix) ───────────

print(f'\n{"="*60}\nEXP F: DistilRoBERTa Independent Eval')
print('  NOTE: This version has a column name bug.')
print('  Use cell19_expf_roberta.py for the corrected version.')
try:
    from transformers import pipeline
    roberta_pipe = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', device=-1, top_k=None)
    t18_path = f'{DRIVE}/task18_part2_final.csv'
    if os.path.exists(t18_path):
        df_rob = pd.read_csv(t18_path)
        df_rob = df_rob[df_rob['temperature'] == EVAL_TEMP].copy()
    else:
        print('  task18_part2_final.csv not found — using simplex_results_T07.csv')
        df_rob = pd.read_csv(f'{DRIVE}/simplex_results_T07.csv').copy()
    LABEL_MAP = {'humor': 'joy', 'melancholy': 'sadness', 'horror': 'fear'}
    def get_roberta_score(text, target_label):
        try:
            res = roberta_pipe(str(text)[:512], truncation=True)[0]
            return float({r['label'].lower(): r['score'] for r in res}.get(target_label, 0.0))
        except: return 0.0
    rob_scores = []
    for _, row in df_rob.iterrows():
        em = row['emotion_target']; target = LABEL_MAP.get(em, 'joy')
        text = str(row.get('generated_text', row.get('output', row.get('steered_text', ''))))[:300]
        rob_scores.append(get_roberta_score(text, target))
    df_rob['roberta_target_score'] = rob_scores
    df_rob.to_csv(f'{DRIVE}/task_roberta_eval.csv', index=False)
    exp_f_results = {}
    for emotion in EMOTIONS:
        sub = df_rob[df_rob['emotion_target'] == emotion]
        shift_col = 'ps_shift' if 'ps_shift' in sub.columns else 'score_humor'
        r, p = scipy_stats.spearmanr(sub[shift_col], sub['roberta_target_score'])
        sig  = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        rob_acc = (sub['roberta_target_score'] > 0.3).mean()
        print(f'  {emotion:12}: r={r:+.3f}  p={p:.4f} {sig}  RoBERTa>0.3: {rob_acc:.1%}')
        exp_f_results[emotion] = {'spearman_r': float(r), 'p': float(p), 'roberta_acc': float(rob_acc)}
    results_all['exp_f'] = exp_f_results
except Exception as e:
    print(f'  DistilRoBERTa failed: {e}')
    results_all['exp_f'] = {'error': str(e)}


# ── Save master summary ───────────────────────────────────────────

with open(f'{DRIVE}/cell12_summary.json', 'w') as f:
    json.dump(results_all, f, indent=2, default=str)

print(f'\n━━━ CELL 17 COMPLETE ━━━')
print('  Run cell18 to fix JSON key types, then cell19 for corrected Exp F.')
