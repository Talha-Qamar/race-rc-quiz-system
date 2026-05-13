# Contributing Guide

Thanks for your interest in improving this project.

## Setup

1. Fork the repository.
2. Create a feature branch.
3. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development Principles

- Keep code changes scoped and focused.
- Prefer reproducible scripts under `src/`.
- Do not commit local datasets or trained model binaries.
- Document any behavior or metric changes in your PR description.

## Pull Requests

- Use clear commit messages.
- Include a concise summary of what changed and why.
- Include instructions to reproduce training/inference behavior if relevant.
