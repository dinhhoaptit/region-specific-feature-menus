"""Probe OpenML candidates for hard large-p / large-n regression."""

from __future__ import annotations

import socket
import warnings

import numpy as np
from sklearn.datasets import fetch_openml

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(20)

CANDIDATES = [
    # (tag, name, data_id)
    ("mercedes", "Mercedes_Benz_Greener_Manufacturing", None),
    ("yolanda", "Yolanda", None),
    ("buzz", "Buzzinsocialmedia_Twitter", None),
    ("yearmsd", "YearPredictionMSD", None),
    ("onlinenews", "OnlineNewsPopularity", None),
    ("allstate", "Allstate_Claims_Severity", None),
    ("blackfriday", "black_friday", None),
    ("diamonds", "diamonds", None),
    ("nyc_taxi", "nyc-taxi-green-dec-2016", None),
    ("topo", "topo_2_1", None),
    ("superconduct", "superconduct", None),
    ("tecator", None, 505),
    ("qsar45077", None, 45077),
    ("fri100", "fri_c4_500_100", None),
    ("fri50", "fri_c3_1000_50", None),
    ("pol", "pol", None),
    ("ailerons", "Ailerons", None),
    ("elevators", "elevators", None),
    ("house16", "house_16H", None),
    ("cpu_act", "cpu_act", None),
    ("sulfur", "sulfur", None),
    ("brazilian_houses", "Brazilian_houses", None),
    ("fps", "fps-in-video-games", None),
    ("wave_energy", "wave_energy", None),
    ("grid", "GridStability", None),
    ("id_42225", None, 42225),
    ("id_42705", None, 42705),
    ("id_42712", None, 42712),
    ("id_42713", None, 42713),
    ("id_42714", None, 42714),
    ("id_42715", None, 42715),
    ("id_42716", None, 42716),
    ("id_42717", None, 42717),
    ("id_42718", None, 42718),
    ("id_42719", None, 42719),
    ("id_42720", None, 42720),
    ("id_42726", None, 42726),
    ("id_42727", None, 42727),
    ("id_42728", None, 42728),
    ("id_42729", None, 42729),
    ("id_42730", None, 42730),
    ("id_41540", None, 41540),
    ("id_41444", None, 41444),
    ("id_42572", None, 42572),
]


def main() -> None:
    ok = []
    for tag, name, did in CANDIDATES:
        print(f"TRY {tag}", flush=True)
        try:
            if did is not None:
                d = fetch_openml(
                    data_id=did, as_frame=False, parser="liac-arff", n_retries=1
                )
            else:
                d = fetch_openml(
                    name=name, as_frame=False, parser="liac-arff", n_retries=1
                )
            X = np.asarray(d.data)
            if hasattr(X, "toarray"):
                X = X.toarray()
            y = np.asarray(d.target)
            # multi-target: take first column
            if y.ndim > 1:
                y = y[:, 0]
            nuniq = len(np.unique(y.astype(str)))
            print(
                f"  OK n={X.shape[0]} p={X.shape[1]} yuniq={nuniq} "
                f"target={d.target_names}",
                flush=True,
            )
            ok.append((tag, name, did, X.shape[0], X.shape[1], nuniq))
        except Exception as exc:
            print(f"  FAIL {type(exc).__name__}: {exc}", flush=True)

    print("\n=== SUCCESS SUMMARY ===", flush=True)
    for row in sorted(ok, key=lambda r: (-r[4], -r[3])):
        print(row, flush=True)


if __name__ == "__main__":
    main()
