# cell08_override_humor_config.py
#
# Manual override of humor config to L18×12.0.
# The layer sweep produced a different best; this corrects it based on
# the full multiplier sweep evidence.
# Writes the corrected config back to best_configs.json.
#
# Requires: best_configs.json on disk, DRIVE in scope

import json

with open(f'{DRIVE}/best_configs.json') as f:
    CONFIGS = json.load(f)

CONFIGS['humor']['layer'] = 18
CONFIGS['humor']['mult']  = 12.0

with open(f'{DRIVE}/best_configs.json', 'w') as f:
    json.dump(CONFIGS, f, indent=2)

print(CONFIGS)
print('✅ humor config overridden to L18×12.0')
