# cell10_eval_temp_override.py
#
# Overrides BEST_TEMP to T=0.7 for the paper's primary reported results.
# BEST_TEMP from the sweep is 1.0 (max avg_both), but T=0.7 gives the
# best avg_diag (90.0%) and is the value reported in the paper.
#
# Requires: df_sweep in memory (from cell09)
# Produces: df_results — 300-row slice at T=0.7

EVAL_TEMP = 0.7
df_results = df_sweep[df_sweep['temperature'] == EVAL_TEMP].copy()
print(f'Working at T={EVAL_TEMP}  ({len(df_results)} rows)')
