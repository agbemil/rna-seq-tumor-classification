# Data

This project uses the UCI Machine Learning Repository's
**Gene Expression Cancer RNA-Seq** dataset (dataset ID 401).

Official characteristics:

- 801 samples
- 20,531 RNA-Seq expression features
- BRCA, COAD, KIRC, LUAD, and PRAD tumor classes
- DOI: 10.24432/C5R88H
- Dataset license: CC BY 4.0

## Recommended route

```bash
python -m src.run_analysis --source uci
```

This uses `ucimlrepo` to fetch dataset 401.

## Local files

Alternatively, put the official files here:

```text
data/data.csv
data/labels.csv
```

and run:

```bash
python -m src.run_analysis --source local
```

Raw CSV files are excluded from Git.

## Data integrity

The local copy supplied during project refactoring contains 16,383
features rather than the official 20,531. It is therefore not included
as the canonical public dataset.
