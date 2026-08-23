"""Discover high-p OpenML regression datasets."""

from __future__ import annotations

import json
import urllib.request
import warnings

import numpy as np
from sklearn.datasets import fetch_openml

warnings.filterwarnings("ignore")


def search(name_prefix: str, limit: int = 30) -> list[int]:
    url = (
        "https://www.openml.org/api/v1/json/data/list/"
        f"data_name/{name_prefix}/status/active/limit/{limit}"
    )
    ids: list[int] = []
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        ds = data["data"]["dataset"]
        if isinstance(ds, dict):
            ds = [ds]
        for d in ds:
            print("found", d.get("did"), d.get("name"))
            ids.append(int(d["did"]))
    except Exception as exc:
        print("search fail", name_prefix, type(exc).__name__, exc)
    return ids


def main() -> None:
    ids = [505, 851, 45077, 42726, 42727, 42728, 42729, 42730, 42572, 41444]
    for prefix in ["QSAR-TID", "MIP", "tecator", "riboflavin", "TomsHardware", "Buzz"]:
        ids.extend(search(prefix))

    seen: set[int] = set()
    print("--- shapes ---")
    for did in ids:
        if did in seen:
            continue
        seen.add(did)
        try:
            d = fetch_openml(data_id=did, as_frame=False, parser="liac-arff")
            X = np.asarray(d.data)
            y = np.asarray(d.target)
            if hasattr(X, "toarray"):
                X = X.toarray()
            nuniq = len(np.unique(y.astype(str)))
            print(
                f"id={did} name={getattr(d, 'DESCR', '')[:0]} "
                f"n={X.shape[0]} p={X.shape[1]} y_unique={nuniq} "
                f"target={d.target_names}"
            )
            # also print details name if present
            details = getattr(d, "details", {}) or {}
            print("  details_name=", details.get("name"), "default_target=", details.get("default_target_attribute"))
        except Exception as exc:
            print(f"id={did} FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
