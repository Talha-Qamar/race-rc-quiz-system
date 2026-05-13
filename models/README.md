# Model Artifacts Policy

This directory is for local model artifacts and placeholders.

## Expected Local Layout

```text
models/
  model_a/
    traditional/
    neural/
    unsupervised/
  model_b/
    traditional/
    neural/
```

## What Should Stay Out of Git

- Trained model binaries (`.joblib`, `.pkl`, `.pt`, `.h5`)
- Calibration exports and large diagnostic files
- Large generated experiment outputs

## Recommended Publishing Options

1. GitHub Releases for stable snapshots
2. Hugging Face Hub for model distribution
3. Cloud object storage (S3/Blob/GCS) for team workflows
4. DVC for versioned model-data coupling

## Good Practice

- Keep a lightweight model card (`model_card.md`) in Git.
- Add download links + checksums in release notes.
- Version artifacts with semantic tags (`v1.0.0-model-a-rf`).
