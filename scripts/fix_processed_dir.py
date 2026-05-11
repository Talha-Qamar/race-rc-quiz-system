import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json
import csv
import sys

ROOT = Path('data/processed')
if not ROOT.exists():
    print('data/processed not found')
    sys.exit(1)

# 1. Rename dev -> val for filenames
for p in list(ROOT.glob('*dev*')):
    new_name = str(p).replace('dev', 'val')
    print('Renaming', p, '->', new_name)
    shutil.move(str(p), new_name)

# 2. Rename model_a_*.csv to model_a_*_features.csv if needed
for p in ROOT.glob('model_a_*.csv'):
    if p.name.endswith('_features.csv'):
        continue
    parts = p.name.split('.')
    name = parts[0]
    if name.endswith('_features'):
        continue
    new = ROOT / f"{name}_features.csv"
    print('Renaming', p.name, '->', new.name)
    shutil.move(str(p), str(new))

# 3. Ensure model_a_*_X.npz dev->val
for p in list(ROOT.glob('model_a_*_X.npz')):
    if 'dev' in p.name:
        new = ROOT / p.name.replace('dev','val')
        print('Renaming', p.name, '->', new.name)
        shutil.move(str(p), str(new))

# 4. Rename model_b dev->val
for p in list(ROOT.glob('model_b_*dev*')):
    new = ROOT / p.name.replace('dev','val')
    print('Renaming', p.name, '->', new.name)
    shutil.move(str(p), str(new))

# 5. Drop *_tokens columns from CSVs safely (stream using pandas with usecols)
csvs = list(ROOT.glob('*.csv'))
for csvf in csvs:
    print('Processing tokens drop for', csvf.name)
    # read header
    try:
        with open(csvf, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            header = next(reader)
    except Exception as e:
        print('Failed to read header for', csvf.name, 'error:', e)
        continue
    keep = [c for c in header if not c.endswith('_tokens')]
    # Read only keep columns
    try:
        df = pd.read_csv(csvf, usecols=keep)
        tmp = csvf.with_suffix('.tmp')
        df.to_csv(tmp, index=False)
        shutil.move(str(tmp), str(csvf))
    except Exception as e:
        print('Failed to process', csvf.name, 'error:', e)

# 6. Create y_*.npy from model_a_*_features.csv label column
for split in ['train','val','test']:
    f = ROOT / f'model_a_{split}_features.csv'
    if not f.exists():
        print('Missing', f)
        continue
    print('Extracting labels to y_', split)
    try:
        # read only label column
        y = pd.read_csv(f, usecols=['label'])['label'].to_numpy(dtype=np.int8)
        np.save(str(ROOT / f'y_{split}.npy'), y)
    except Exception as e:
        print('Failed to extract labels for', f, 'error:', e)

# 7. Update manifest keys dev->val
mf = ROOT / 'preprocessing_manifest.json'
if mf.exists():
    print('Updating manifest', mf)
    data = json.loads(mf.read_text(encoding='utf-8'))
    # replace keys in files
    new_files = {}
    for k,v in data.get('files',{}).items():
        new_k = k.replace('dev','val')
        new_v = v.replace('dev','val') if isinstance(v,str) else v
        new_files[new_k] = new_v
    data['files'] = new_files
    # replace split keys
    if 'splits' in data:
        new_splits = {}
        for k,v in data['splits'].items():
            new_k = k.replace('dev','val')
            new_splits[new_k] = v
        data['splits'] = new_splits
    mf.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    print('Manifest updated')
else:
    print('No manifest to update')

print('Done')
