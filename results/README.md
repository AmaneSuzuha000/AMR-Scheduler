# results — experiment data

34 official CSV files, 4872 rows total. These are the authoritative benchmark
and validation results.

| Path | Description |
|---|---|
| `MANIFEST.json` | sha256, row count, column names, and per-column means of all 34 CSVs |
| `*.csv` | Official experiment and benchmark datasets |
| `figures/` | Rendered figures and plots |
| `replay.html` | Self-contained process replay — open it in a browser |
| `archive/` | Historical reference data |

## Validation

```bash
python tools/data_manifest.py --check    # detect silent data drift
python tools/reproduce.py --stage quick  # re-run and diff every field
```

## Data/code ordering discipline

Official data must be newer than the code that produced it. If you change
`amr_sched/`, re-run the affected experiments and refresh `MANIFEST.json`;
otherwise `data_manifest.py --check` will report a mismatch.
