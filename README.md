# Region-specific feature menus

Python implementation for the manuscript

**Region-Specific Feature Menus for Progressive Feature Revelation**

(Hoa Dinh Nguyen, Posts and Telecommunications Institute of Technology).

This repository provides the code used to produce the reported anytime RMSE tables and figures. The method partitions the input space, learns an ordered forward-stepwise feature menu in each region, and predicts at every feature budget with nested prefix least-squares models.

**Repository:** https://github.com/dinhhoaptit/region-specific-feature-menus

## Install

```text
pip install -r requirements.txt
```

Requires Python 3.10+ (NumPy, SciPy, scikit-learn). Matplotlib is needed only for figure scripts.

## Data

Public datasets are downloaded once into `data/` (not stored in this repository):

```text
python examples/download_datasets.py
```

See `data/README.md`. Experiments then run offline from `data/*.npz`.

| Dataset | Archive |
|---|---|
| CT slices | UCI, DOI [10.24432/C55C8Z](https://doi.org/10.24432/C55C8Z) |
| Superconductivity | UCI, DOI [10.24432/C53P47](https://doi.org/10.24432/C53P47) |
| tecator | OpenML 505 |
| topo_2_1 | OpenML 422 |

## Reproduce paper experiments

```text
python examples/extra_evidence.py
python examples/stability_budget.py
python examples/conditional_greedy.py --ct --sc --max-rows 8000
python examples/when_to_regionalize.py --all --max-rows 8000
python examples/named_menus.py
python examples/modern_baselines.py
python examples/generate_paper_figures.py
```

JSON outputs and manuscript packages are written locally under `submission_*/` (not tracked in this repository). Figure PDFs can be regenerated with `examples/generate_paper_figures.py` and `examples/stability_budget.py`.

## Layout

| Path | Contents |
|---|---|
| `src/` | Library: region menus, selectors (stepwise, VIF, Lasso–LARS, SAOLA-style, mRMR), progressive evaluation, LRG |
| `examples/` | Dataset loaders and paper experiment scripts |
| `data/` | Local downloads only (ignored by git) |

## License / contact

Research code accompanying the manuscript. Contact Hoa Dinh Nguyen (`hoand@ptit.edu.vn`) for reuse questions.
