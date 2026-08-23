# Code availability (KAIS submission)

Public repository: https://github.com/dinhhoaptit/region-specific-feature-menus

Implementation is at the **project root**, not only in this folder.

```text
src/region_menus.py          partition, menus, prefix models, nested pre-test helpers
src/feature_catalog.py       UCI CT and superconductivity column names
src/conditional_greedy.py    local residual-greedy (LRG) baseline
src/regionalize.py           train-only global vs regional choice
src/selectors.py             forward stepwise, Lasso-LARS, VIF
src/vif.py                   streamwise VIF / alpha-investing
src/progressive.py           anytime RMSE evaluation
examples/data_loaders.py     local UCI/OpenML npz loaders
examples/generate_paper_figures.py
examples/kais_extra_evidence.py
examples/kais_stability_budget.py
examples/kais_conditional_greedy.py
examples/kais_when_to_regionalize.py
examples/kais_named_menus.py
requirements.txt
README.md
data/README.md
```

## Reproduce

From the project root:

```text
pip install -r requirements.txt
python examples/download_datasets.py
python examples/kais_extra_evidence.py
python examples/kais_stability_budget.py
python examples/kais_conditional_greedy.py --ct --sc --max-rows 8000
python examples/kais_when_to_regionalize.py --all --max-rows 8000
python examples/kais_named_menus.py
```

Public datasets are stored as local `data/*.npz` after the download script. No network is required at experiment time.

## JSON that supports the tables

| File | Tables / figures |
|---|---|
| `extra_evidence.json` | Localized Lasso–LARS; menu Jaccard |
| `named_menus.json` | Table of named prefixes |
| `conditional_greedy.json` | LRG anytime RMSE |
| `when_to_regionalize.json` | Nested global vs regional pre-test |
| `stability_and_budget.json` | Stability table; measurement-budget figure |
