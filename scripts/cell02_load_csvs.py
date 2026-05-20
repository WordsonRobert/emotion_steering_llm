# cell02_load_csvs.py
#
# Loads and validates the three master CSVs.
# Auto-detects the text column and drops malformed rows.
#
# Requires:
#   DRIVE (from cell00)
#
# Produces (in memory):
#   df_humor, df_mel, df_hor
#   dark_texts  — mel + horror combined list
#   N_HUMOR, N_MEL, N_HOR, N_DARK
#
# Runtime: <1 min

def _load_and_clean(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert len(df) > 0, f'{label}: CSV is empty'

    if 'text' in df.columns:
        text_col = 'text'
    else:
        candidates = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ['text', 'content', 'body', 'post'])
        ]
        assert candidates, (
            f'{label}: No text column found. Columns: {df.columns.tolist()}'
        )
        text_col = candidates[0]
        df = df.rename(columns={text_col: 'text'})
        print(f'   {label}: renamed column "{text_col}" → "text"')

    df['text'] = df['text'].fillna('').astype(str).str.strip()
    before = len(df)
    df = df[df['text'].str.len() > 10].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f'   {label}: dropped {dropped} rows with text len ≤ 10')

    return df


print('Loading CSVs...')
df_humor = _load_and_clean(f'{DRIVE}/humor_master.csv',      'humor')
df_mel   = _load_and_clean(f'{DRIVE}/melancholy_master.csv', 'melancholy')
df_hor   = _load_and_clean(f'{DRIVE}/horror_master.csv',     'horror')

N_HUMOR = len(df_humor)
N_MEL   = len(df_mel)
N_HOR   = len(df_hor)

dark_texts = df_mel['text'].tolist() + df_hor['text'].tolist()
N_DARK     = len(dark_texts)

print(f'\n✅ Datasets loaded')
print(f'   humor:      {N_HUMOR} samples')
print(f'   melancholy: {N_MEL} samples')
print(f'   horror:     {N_HOR} samples')
print(f'   dark total: {N_DARK} (mel + horror combined)')

print('\nText length stats (chars):')
for name, df in [('humor', df_humor), ('melancholy', df_mel), ('horror', df_hor)]:
    lengths = df['text'].str.len()
    print(f'   {name:12} min={lengths.min()}  median={lengths.median():.0f}  max={lengths.max()}')

print('\nSample texts (first row each):')
for name, df in [('humor', df_humor), ('melancholy', df_mel), ('horror', df_hor)]:
    print(f'   [{name}] {df["text"].iloc[0][:120]}')

print('\n✅ Cell 2 complete')
