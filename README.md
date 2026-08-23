# Region-specific incremental feature menus

Python implementation for the manuscript *Region-Specific Incremental Feature Menus for Progressive Feature Revelation*, prepared as a regular article for *Knowledge and Information Systems*.

The method partitions the input space, learns an ordered forward-stepwise feature menu in each region, and predicts at every feature budget with nested prefix least-squares models.

## Install

```text
pip install -r requirements.txt
```

Requires Python 3.10+ and the packages in `requirements.txt` (NumPy, SciPy, scikit-learn). Matplotlib is needed only for figure scripts.

## Data

Public datasets are downloaded once into `data/`:

```text
python examples/download_datasets.py
```

See `data/README.md`. Experiments then run offline from `data/*.npz`.

## Reproduce the paper artifacts

```text
python examples/kais_extra_evidence.py
python examples/generate_paper_figures.py
```

- `examples/kais_extra_evidence.py` writes `submission_kais/extra_evidence.json` (regional menu prefixes, Jaccard overlap, localized Lasso–LARS anytime RMSE).
- `examples/kais_stability_budget.py` writes `submission_kais/stability_and_budget.json` and `fig6_measurement_budget.pdf` (measurement budget to target error; menu stability across splits).
- `examples/generate_paper_figures.py` writes PDF/PNG figures. By default it currently targets `submission_kbs/figures/`; copy or point the output directory to `submission_kais/figures/` if regenerating the KAIS figures in place.

## Layout

| Path | Contents |
|---|---|
| `src/` | Library: menus, selectors, progressive evaluation |
| `examples/` | Loaders, demos, paper experiments |
| `submission_kais/` | KAIS manuscript, figures, cover letter |
| `submission_kbs/` | Previous *Knowledge-Based Systems* package (do not mix) |

## License

Research code accompanying the manuscript. Contact Hoa Dinh Nguyen (`hoand@ptit.edu.vn`) for reuse questions until a public repository URL is attached to the paper.
