# data/

Place the three master CSV files here before running any scripts.

Each file must contain a `text` column with cleaned, stripped strings.
Length range: 55–199 characters per sample.

| File | Samples | Source |
|---|---|---|
| `humor_master.csv` | 500 | Short jokes (Groq-compressed) |
| `melancholy_master.csv` | 419 | r/depression posts (Groq-compressed) |
| `horror_master.csv` | 199 | r/nosleep posts (Groq-compressed) |

These files are NOT committed to the repo. They are the output of a manual
curation + Groq API compression pipeline run prior to this experiment.

The pipeline reads from this folder via:
    DRIVE = '/content/drive/MyDrive/STEERING_EMNLP_2026'

So in Colab, these CSVs live in that Google Drive folder.
The scripts auto-discover the folder — see cell00 for the discovery logic.
