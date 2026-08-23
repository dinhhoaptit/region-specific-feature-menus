# Code availability (KAIS submission)

The implementation lives at the **project root**, not only in this folder.

```text
src/region_menus.py      partition, menus, prefix models, prediction
src/selectors.py         forward stepwise, Lasso-LARS, VIF
src/vif.py               streamwise VIF / alpha-investing
src/progressive.py       anytime RMSE evaluation
examples/data_loaders.py local UCI/OpenML npz loaders
examples/generate_paper_figures.py  main figures
examples/kais_extra_evidence.py     menu overlap + localized Lasso
requirements.txt
README.md
```

## Reproduce

From the project root:

```text
pip install -r requirements.txt
python examples/download_datasets.py
python examples/kais_extra_evidence.py
python examples/kais_stability_budget.py
```

Public datasets are stored as local `data/*.npz` after the download script. No network is required at experiment time.

`submission_kais/extra_evidence.json` supports the menu-overlap and localized-Lasso tables. `submission_kais/stability_and_budget.json` supports the measurement-budget figure and the stability table.
