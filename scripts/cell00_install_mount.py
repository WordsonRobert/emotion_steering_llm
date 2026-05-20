# cell00_install_mount.py
#
# Installs all packages, mounts Google Drive, and auto-discovers the
# STEERING_EMNLP_2026 folder containing the three master CSVs.
# Also handles HuggingFace login via Colab Secrets.
#
# Requires:
#   - Google Colab runtime (A100 recommended)
#   - Google Drive with a folder containing humor_master.csv,
#     melancholy_master.csv, horror_master.csv
#   - HF_TOKEN set in Colab Secrets (Tools → Secrets)
#
# Produces:
#   DRIVE  — string path to the STEERING_EMNLP_2026 folder
#
# Runtime: ~3 min

import subprocess, sys

def _pip(*args):
    subprocess.check_call(
        [sys.executable, '-m', 'pip'] + list(args),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

print('Installing packages...')
_pip('install', 'numpy==2.0.2', '--force-reinstall', '-q')
_pip('install', 'transformer_lens', '--no-deps', '-q')
_pip('install',
     'better-abc==0.0.3',
     'transformers-stream-generator==0.0.5',
     'beartype==0.14.1',
     'huggingface-hub>=0.23.2,<1.0',
     'einops', 'jaxtyping', 'fancy-einsum', '-q')
_pip('install',
     'scikit-learn', 'matplotlib', 'seaborn',
     'scipy', 'datasets', 'transformers', '-q')
print('✅ Packages installed')

# ── Mount Google Drive ────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ── Auto-discover STEERING_EMNLP_2026 folder ─────────────────────
import os

REQUIRED_CSVS = [
    'humor_master.csv',
    'melancholy_master.csv',
    'horror_master.csv',
]

SEARCH_ROOTS = [
    '/content/drive/MyDrive',
    '/content/drive/Shareddrives',
]

def _has_csvs(path):
    """Return True if path contains all three required CSVs."""
    return all(os.path.exists(os.path.join(path, f)) for f in REQUIRED_CSVS)

DRIVE = None

# First: exact folder name match
for root in SEARCH_ROOTS:
    if not os.path.exists(root):
        continue
    candidate = os.path.join(root, 'STEERING_EMNLP_2026')
    if os.path.exists(candidate) and _has_csvs(candidate):
        DRIVE = candidate
        break

# Second: shallow walk (one level deep) looking for any folder with the 3 CSVs
if DRIVE is None:
    for root in SEARCH_ROOTS:
        if not os.path.exists(root):
            continue
        try:
            for entry in os.scandir(root):
                if entry.is_dir() and _has_csvs(entry.path):
                    DRIVE = entry.path
                    break
        except PermissionError:
            pass
        if DRIVE:
            break

# Fallback: manual input
if DRIVE is None:
    print(
        '⚠️  Could not auto-discover the folder.\n'
        'Paste the full path to the folder that contains '
        'humor_master.csv / melancholy_master.csv / horror_master.csv:'
    )
    DRIVE = input('DRIVE path: ').strip().rstrip('/')
    assert _has_csvs(DRIVE), (
        f'Folder {DRIVE} does not contain all three required CSVs. '
        f'Found: {os.listdir(DRIVE)}'
    )

print(f'✅ DRIVE = {DRIVE}')
print('   Contents:', sorted(os.listdir(DRIVE)))

# ── HuggingFace login ─────────────────────────────────────────────
HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    pass

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
    print('✅ HuggingFace login successful')
else:
    print(
        '⚠️  HF_TOKEN not found in Colab Secrets.\n'
        '   Go to Tools → Secrets → Add new secret → Name: HF_TOKEN\n'
        '   Then re-run this cell.'
    )

print('\n✅ Cell 0 complete')
