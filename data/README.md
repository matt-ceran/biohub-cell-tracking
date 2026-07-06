# Data Directory

Place Kaggle competition data under `data/raw/`.
Keep raw downloaded data out of Git.

Expected layout after download:

```text
data/raw/
  train/
    sample_name.zarr/
    sample_name.geff/
  test/
    sample_name.zarr/
  sample_submission.csv
```

Use `data/working/` for derived local artifacts that can be regenerated.

After Kaggle authentication and rules acceptance, run:

```bash
python scripts/download_data.py
```
