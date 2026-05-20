# cell16_tasks_2_6.py
#
# Task 2: Additive vs Projection-Injection hook comparison.
#   Same prompts, same vectors, same layer/mult from CONFIGS.
#   n=30 per emotion, T=0.7.
#   RESULT: horror proj=63% vs add=90% (**); humor/mel ns.
#
# Task 6: 2-Turn Tonal Drift.
#   Horror-steered turn 1 → neutral follow-up turn 2 (no steering).
#   30 conversation pairs. T=0.7.
#   RESULT: T1=+5.855 → T2=+7.729; drift=−1.874; p=0.042*; 23.3% neutralized.
#
# Requires (from cell15):
#   model, device, EMOTIONS, PROBE_LAYER, DRIVE, CONFIGS
#   df_humor, df_mel, df_hor, sv
#   PROBE_FOR, PROBE_SIGN, coherence_score, probe_score, build_hook
#
# Runtime: ~1.5 hrs on A100

import torch, numpy as np, pandas as pd, json, os, math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from collections import Counter
from tqdm import tqdm

DRIVE     = '/content/drive/MyDrive/STEERING_EMNLP_2026'
BEST_TEMP = 0.7
actual_t  = 1e-10 if BEST_TEMP == 0.0 else BEST_TEMP

print(f'CELL 16: Tasks 2 + 6  |  BEST_TEMP={BEST_TEMP}')


# ── Hook factories ────────────────────────────────────────────────

def make_proj_inject_hook(svec, mult):
    def fn(resid_pre, hook):
        proj = (resid_pre @ svec).unsqueeze(-1) * svec
        return (resid_pre - proj) + svec * mult
    return fn

def make_additive_hook(svec, mult):
    def fn(resid_pre, hook):
        return resid_pre + svec * mult
    return fn

def get_hooks(emotion, hook_factory):
    cfg = CONFIGS[emotion]; svec = sv[emotion]
    mult = float(cfg['mult']); layer = int(cfg['layer'])
    return [(f'blocks.{layer}.hook_resid_pre', hook_factory(svec, mult))]


# ── Task 2: Additive vs Proj-Inject ──────────────────────────────

print(f'\n{"="*60}')
print('TASK 2: Additive vs Projection-Injection')
print(f'  n=30 per emotion  T={BEST_TEMP}')

TASK2_N = 30
task2_prompts = {
    em: df.sample(TASK2_N, random_state=99).reset_index(drop=True)['text'].tolist()
    for em, df in [('humor', df_humor), ('melancholy', df_mel), ('horror', df_hor)]
}

def run_task2_eval(emotion, hook_factory, mode_name, prompts):
    hooks = get_hooks(emotion, hook_factory)
    w     = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
    rows  = []
    for prompt in prompts:
        p = prompt[:120]
        base_out = model.generate(p, max_new_tokens=50, temperature=actual_t, verbose=False)
        base_ps  = probe_score(base_out, w, sign)
        with model.hooks(fwd_hooks=hooks):
            ste_out = model.generate(p, max_new_tokens=50, temperature=actual_t, verbose=False)
        ste_ps = probe_score(ste_out, w, sign)
        coh    = coherence_score(ste_out[len(p):].strip())
        rows.append({'emotion': emotion, 'mode': mode_name,
                     'ps_shift': float(ste_ps - base_ps), 'pred_correct': bool(ste_ps > 0),
                     'coherent': bool(coh >= 0.6), 'both': bool((ste_ps > base_ps) and (coh >= 0.6)),
                     'coherence': float(coh)})
    return rows

task2_rows = []
for emotion in EMOTIONS:
    prompts = task2_prompts[emotion]
    print(f'\n  {emotion.upper()}:')
    for mode_name, factory in [('proj_inject', make_proj_inject_hook), ('additive', make_additive_hook)]:
        print(f'    {mode_name} ...', end='', flush=True)
        rows = run_task2_eval(emotion, factory, mode_name, prompts)
        task2_rows.extend(rows)
        both = np.mean([r['both'] for r in rows])
        print(f'  both={both:.1%}  shift={np.mean([r["ps_shift"] for r in rows]):+.3f}')

df_task2 = pd.DataFrame(task2_rows)
df_task2.to_csv(f'{DRIVE}/task2_additive_vs_projinject.csv', index=False)

print(f'\n{"="*60}')
print('TASK 2 STATS (Mann-Whitney U, proj_inject vs additive)')
task2_stats = {}
for emotion in EMOTIONS:
    sub = df_task2[df_task2['emotion'] == emotion]
    pi  = sub[sub['mode'] == 'proj_inject']['ps_shift'].values
    ad  = sub[sub['mode'] == 'additive']['ps_shift'].values
    u, p = scipy_stats.mannwhitneyu(pi, ad, alternative='two-sided')
    sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    pi_both = sub[sub['mode']=='proj_inject']['both'].mean()
    ad_both = sub[sub['mode']=='additive']['both'].mean()
    print(f'  {emotion:12}: proj={pi_both:.1%}  add={ad_both:.1%}  U={u:.0f}  p={p:.4f}  {sig}')
    task2_stats[emotion] = {'U': float(u), 'p': float(p), 'sig': sig,
                            'proj_both': float(pi_both), 'add_both': float(ad_both)}
print('✅ task2_additive_vs_projinject.csv saved')


# ── Task 6: 2-Turn Tonal Drift ────────────────────────────────────

print(f'\n{"="*60}')
print('TASK 6: 2-Turn Tonal Drift')
print(f'  30 horror-steered turn1 + neutral turn2  T={BEST_TEMP}')

DRIFT_N = 30
drift_prompts   = df_hor.sample(DRIFT_N, random_state=55).reset_index(drop=True)['text'].tolist()
neutral_followups = ["What happened next?", "Tell me more.", "Continue the story.", "And then?",
                     "What did you do after that?"] * 10
horror_hooks    = get_hooks('horror', make_proj_inject_hook)

def get_all_scores(text):
    tokens = model.to_tokens([text], prepend_bos=True)
    if tokens.shape[1] > 512: tokens = tokens[:, -512:]
    hn = f'blocks.{PROBE_LAYER}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: hn in n, stop_at_layer=PROBE_LAYER+1)
    act = cache[hn][0, -1, :].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return {'score_humor': float(act @ w_hm), 'score_mel': float(act @ (-w_hm)), 'score_horror': float(act @ (-w_hh))}

drift_rows = []
for i, prompt in enumerate(tqdm(drift_prompts, desc='Drift pairs')):
    p = prompt[:120]; followup = neutral_followups[i % len(neutral_followups)]
    with model.hooks(fwd_hooks=horror_hooks):
        turn1_out = model.generate(p, max_new_tokens=60, temperature=actual_t, verbose=False)
    t1_scores = get_all_scores(turn1_out)
    turn2_ctx = turn1_out + ' ' + followup
    if len(model.to_tokens([turn2_ctx])[0]) > 400:
        turn2_ctx = turn1_out[-200:] + ' ' + followup
    turn2_out = model.generate(turn2_ctx, max_new_tokens=60, temperature=actual_t, verbose=False)
    t2_scores = get_all_scores(turn2_out)
    drift_rows.append({'pair_id': i, 'prompt': p[:60],
                       'turn1_horror': t1_scores['score_horror'], 'turn2_horror': t2_scores['score_horror'],
                       'horror_drift': t1_scores['score_horror'] - t2_scores['score_horror'],
                       'turn1_gen': turn1_out[len(p):].strip()[:100],
                       'turn2_gen': turn2_out[len(turn2_ctx):].strip()[:100]})

df_drift   = pd.DataFrame(drift_rows)
mean_t1    = df_drift['turn1_horror'].mean()
mean_t2    = df_drift['turn2_horror'].mean()
drift      = df_drift['horror_drift'].mean()
t_stat, t_p = scipy_stats.ttest_rel(df_drift['turn1_horror'].values, df_drift['turn2_horror'].values)
u_stat, u_p = scipy_stats.mannwhitneyu(df_drift['turn1_horror'].values, df_drift['turn2_horror'].values, alternative='greater')
neutralized = (df_drift['turn2_horror'] < df_drift['turn1_horror']).mean()

df_drift.to_csv(f'{DRIVE}/task6_tonal_drift.csv', index=False)

print(f'\n{"="*60}')
print('TASK 6 RESULTS')
print(f'  Turn 1 horror score (steered):   {mean_t1:+.4f}')
print(f'  Turn 2 horror score (unsteered): {mean_t2:+.4f}')
print(f'  Mean drift (T1−T2):              {drift:+.4f}')
print(f'  % pairs where T2 < T1:          {neutralized:.1%}')
print(f'  Paired t-test: t={t_stat:.3f}  p={t_p:.4f}')
print(f'  Mann-Whitney:  U={u_stat:.0f}  p={u_p:.4f}')

task6_summary = {'n_pairs': DRIFT_N, 'best_temp': BEST_TEMP,
                 'mean_t1_horror': float(mean_t1), 'mean_t2_horror': float(mean_t2),
                 'mean_drift': float(drift), 'neutralized_pct': float(neutralized),
                 'paired_t': {'t': float(t_stat), 'p': float(t_p)},
                 'mann_whitney': {'U': float(u_stat), 'p': float(u_p)}}
with open(f'{DRIVE}/task6_tonal_drift_summary.json', 'w') as f:
    json.dump(task6_summary, f, indent=2)

cell11_summary = {'task2': task2_stats, 'task6': task6_summary}
with open(f'{DRIVE}/cell11_summary.json', 'w') as f:
    json.dump(cell11_summary, f, indent=2)

print(f'\n✅ Cell 16 complete')
print(f'   Saved: task2_additive_vs_projinject.csv, task6_tonal_drift.csv')
print(f'          task6_tonal_drift_summary.json, cell11_summary.json')
