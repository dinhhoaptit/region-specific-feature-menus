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
python examples/kais_stability_budget.py
python examples/kais_conditional_greedy.py --ct --sc --max-rows 8000
python examples/kais_when_to_regionalize.py --all --max-rows 8000
python examples/kais_named_menus.py
python examples/generate_paper_figures.py
```

JSON records are written under `submission_kais/`. Figure PDFs for the manuscript live in `submission_kais/figures/` (`fig6_measurement_budget.pdf` is produced by `kais_stability_budget.py`). `generate_paper_figures.py` currently writes `fig1`--`fig5` under `submission_kbs/figures/` as well; copy those PDFs into `submission_kais/figures/` if regenerating them.

## Layout

| Path | Contents |
|---|---|
| `src/` | Library: menus, selectors, progressive evaluation |
| `examples/` | Loaders, demos, paper experiments |
| `submission_kais/` | KAIS manuscript, figures, cover letter |
| `submission_kbs/` | Previous *Knowledge-Based Systems* package (do not mix) |

## License

Research code accompanying the manuscript. Contact Hoa Dinh Nguyen (`hoand@ptit.edu.vn`) for reuse questions until a public repository URL is attached to the paper.
