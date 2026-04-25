import pandas as pd
import os
import yaml

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(base_dir, 'data', 'raw')
links = pd.read_csv(os.path.join(raw_dir, 'links.csv'))
metadata_path = os.path.join(raw_dir, 'metadata.csv')

existing = set()
if os.path.exists(metadata_path):
    df = pd.read_csv(metadata_path)
    existing = set(df['movieId'])

new_rows = []
for _, row in links.iterrows():
    if row['movieId'] not in existing:
        new_rows.append({
            'movieId': row['movieId'],
            'overview': 'Great movie overview.',
            'cast': 'Actor A Actor B',
            'director': 'Director X'
        })

if new_rows:
    pd.DataFrame(new_rows).to_csv(metadata_path, mode='a', header=not os.path.exists(metadata_path), index=False)
    print(f"Filled {len(new_rows)} mock metadata rows.")
