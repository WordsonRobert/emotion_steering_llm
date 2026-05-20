# cell18_fix_cell12_summary.py
#
# Fixes cell12_summary.json by converting dict keys to strings
# (json.dump fails on numeric keys in some Python versions).
# Also inspects simplex_results_T07.csv to confirm column names
# before running the corrected Exp F in cell19.
#
# Requires: results_all in memory (from cell17), DRIVE in scope

import json, pandas as pd

results_all_clean = {}
for exp_key, val in results_all.items():
    if isinstance(val, dict):
        results_all_clean[exp_key] = {str(k): v for k, v in val.items()}
    else:
        results_all_clean[exp_key] = val

with open(f'{DRIVE}/cell12_summary.json', 'w') as f:
    json.dump(results_all_clean, f, indent=2)
print('✅ cell12_summary.json saved (string keys)')

df_check = pd.read_csv(f'{DRIVE}/simplex_results_T07.csv')
print(df_check.columns.tolist())
print(df_check.head(2))
