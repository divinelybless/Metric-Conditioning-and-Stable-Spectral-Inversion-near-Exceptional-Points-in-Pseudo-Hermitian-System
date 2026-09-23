# Numerical codes for Metric Conditioning and Stable Spectral Inversion

This repository contains only the numerical/reproducibility code associated with the manuscript:

**Metric Conditioning and Stable Spectral Inversion near Exceptional Points in Pseudo-Hermitian Systems**

## Contents

- `code/reproduce_figures.py` — numerical checks and figure-generation code
- `requirements.txt` — Python dependencies
- `.gitignore` — standard ignored files

## Running the code

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python code/reproduce_figures.py
```

The script checks the main two-level, three-level, and growing-dimensional numerical bounds and generates the figures used in the study.

## Funding

This research received no external funding.

## Authors

Dorcas Attuabea Addo, Richard Kena Boadi, Jeffrey Ezrean, Richard Kwame Ansah, and Peter Yeboah.
