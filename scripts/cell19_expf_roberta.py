# cell19_expf_roberta.py
#
# Corrected Exp F: DistilRoBERTa independent evaluation on steered outputs.
# Fix: uses 'ste_gen' column from simplex_results_T07.csv (correct column).
# Cell17's version used wrong column names ('generated_text', 'output', etc.).
#
# Scores each steered generation with j-hartmann/emotion-english-distilroberta-base.
# Computes Spearman r between probe shift (ps_shift) and RoBERTa target score.
# Cross-validates against DeepSeek judge scores (cell20).
#
# RESULT:
#   humor:  RoBERTa>0.3=29%  r=+0.207 p=0.038*
#   mel:    RoBERTa>0.3=41%  r=+0.041 ns
#   horror: RoBERTa>0.3=53%  r=+0.031 ns
#
# Requires on disk: simplex_results_T07.csv, cell12_summary.json
# Requires packages: transformers (pipeline)
#
# Runtime: ~10 min on CPU

from transformers import pipeline
from scipy import stats as scipy_stats
from tqdm import tqdm
import pandas as pd, json

DRIVE     = '/content/drive/MyDrive/STEERING_EMNLP_2026'
EVAL_TEMP = 0.7
EMOTIONS  = ['humor', 'melancholy', 'horror']
LABEL_MAP = {'humor': 'joy', 'melancholy': 'sadness', 'horror': 'fear'}

roberta_pipe = pipeline(
    'text-classification',
    model='j-hartmann/emotion-english-distilroberta-base',
    device=-1, top_k=None,
)
print('✅ DistilRoBERTa loaded on CPU')

df_rob = pd.read_csv(f'{DRIVE}/simplex_results_T07.csv')
print(f'  {len(df_rob)} rows  columns: {df_rob.columns.tolist()}')


def get_roberta_score(text, target_label):
    try:
        res    = roberta_pipe(str(text)[:512], truncation=True)[0]
        scores = {r['label'].lower(): r['score'] for r in res}
        return float(scores.get(target_label, 0.0))
    except:
        return 0.0


rob_scores = []
for _, row in tqdm(df_rob.iterrows(), total=len(df_rob), desc='  RoBERTa'):
    em     = row['emotion_target']
    target = LABEL_MAP.get(em, 'joy')
    text   = str(row['ste_gen'])[:300]   # correct column
    rob_scores.append(get_roberta_score(text, target))

df_rob['roberta_target_score'] = rob_scores
df_rob.to_csv(f'{DRIVE}/task_roberta_eval.csv', index=False)
print(f'✅ task_roberta_eval.csv saved')

print(f'\n{"="*60}')
print('EXP F RESULTS')
exp_f_results = {}
for emotion in EMOTIONS:
    sub     = df_rob[df_rob['emotion_target'] == emotion]
    rob_acc = (sub['roberta_target_score'] > 0.3).mean()
    r, p    = scipy_stats.spearmanr(sub['ps_shift'], sub['roberta_target_score'])
    sig     = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    print(f'  {emotion:12}: RoBERTa>0.3={rob_acc:.1%}  Spearman r={r:+.3f}  p={p:.4f}  {sig}')
    exp_f_results[emotion] = {'roberta_acc': float(rob_acc), 'spearman_r': float(r), 'p': float(p), 'sig': sig}

print(f'\n  Mean RoBERTa target scores:')
for emotion in EMOTIONS:
    sub = df_rob[df_rob['emotion_target'] == emotion]
    print(f'  {emotion:12}: {sub["roberta_target_score"].mean():.4f} ± {sub["roberta_target_score"].std():.4f}')

try:
    with open(f'{DRIVE}/cell12_summary.json') as f: cell12 = json.load(f)
except: cell12 = {}
cell12['exp_f'] = exp_f_results
with open(f'{DRIVE}/cell12_summary.json', 'w') as f:
    json.dump(cell12, f, indent=2)

print(f'\n✅ cell12_summary.json updated with corrected Exp F')
print(f'━━━ EXP F COMPLETE ━━━')
