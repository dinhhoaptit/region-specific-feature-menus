"""Job (5): name regional menus using UCI feature documentation.

    python examples/paa_named_menus.py

Writes ``submission_paa/named_menus.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.data_loaders import load_ct_slices, load_superconductivity
from examples.generate_paper_figures import prepare_xy
from src.feature_catalog import (
    ct_pretty_name,
    superconductivity_pretty_name,
)
from src.region_menus import fit_region_menus

OUT = ROOT / "submission_paa" / "named_menus.json"
K = 6


def _fit(name, X, y, n_regions, max_features):
    X_train, _, y_train, _ = prepare_xy(X, y, seed=0)
    min_size = max(40, min(200, len(y_train) // (n_regions * 5)))
    return fit_region_menus(
        X_train,
        y_train,
        n_regions=n_regions,
        partition="residual",
        assignment="soft",
        soft_temperature=1.0,
        min_region_size=min_size,
        max_features=max_features,
        selector="forward_stepwise",
        random_state=0,
    )


def named_prefixes(model, pretty_fn, k: int = K):
    rows = []
    for menu in model.menus:
        idx = [int(j) for j in menu.selected[:k]]
        rows.append(
            {
                "region": int(menu.region_id),
                "n_samples": int(menu.n_samples),
                "indices": idx,
                "names": [pretty_fn(j) for j in idx],
            }
        )
        print(
            f"  r={menu.region_id} n={menu.n_samples} "
            + "; ".join(f"{j}:{pretty_fn(j)}" for j in idx),
            flush=True,
        )
    return rows


def main():
    out = {}
    print("[ct]", flush=True)
    X, y = load_ct_slices(max_rows=8000, seed=0)
    ct = _fit("ct", X, y, 4, 25)
    bone_lead = []
    for menu in ct.menus:
        top = menu.selected[:K]
        n_bone = sum(int(j) < 240 for j in top)
        bone_lead.append(n_bone)
    out["ct_slices"] = {
        "prefixes": named_prefixes(ct, ct_pretty_name),
        "n_bone_in_top6": bone_lead,
    }
    print("[superconductivity]", flush=True)
    X, y = load_superconductivity(max_rows=None, seed=0)
    sc = _fit("sc", X, y, 4, 25)
    out["superconductivity"] = {"prefixes": named_prefixes(sc, superconductivity_pretty_name)}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
