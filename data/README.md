# Data Directory Policy

This directory is intentionally excluded from Git for large-file management.

## Expected Local Layout

```text
data/
  raw/
  processed/
```

## What Should Stay Out of Git

- Raw dataset CSV files
- Processed arrays (`.npz`, `.npy`)
- Feature exports and intermediate large files

## Recommended Hosting Options

1. Kaggle Dataset for public reproducibility
2. Google Drive or OneDrive for coursework sharing
3. AWS S3 / Azure Blob for production-style artifact storage
4. DVC remote storage for versioned data pipelines

## Reproducibility

Document and version:
- data source URL
- preprocessing command(s)
- checksum/hash of canonical files
