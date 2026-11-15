# results — experiment data

22 official CSV files, 4560 rows total. These are the source of every number in
the paper.

| Path | Description |
|---|---|
| `MANIFEST.json` | sha256, row count, column names, and per-column means of all 22 CSVs |
| `*.csv` | Official experiment data |
| `figures/` | Paper figures (matplotlib) |
| `replay.html` | Self-contained process replay — open it in a browser |
| `archive/` | **Historical data, not for computation** — see the README inside |

## Validation

```bash
python tools/data_manifest.py --check    # detect silent data drift
python tools/reproduce.py --stage quick  # re-run and diff every field
```

## Data/code ordering discipline

Official data must be newer than the code that produced it. If you change
`amr_sched/`, re-run the affected experiments and refresh `MANIFEST.json`;
otherwise `data_manifest.py --check` will report a mismatch.
