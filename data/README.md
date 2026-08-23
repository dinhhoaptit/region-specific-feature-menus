# Local datasets

Place downloaded experiment data here as compressed NumPy archives:

| File | Source | Approx. size |
|---|---|---|
| `ct_slices.npz` | UCI CT slices (id 206) | \(n\approx 53500\), \(p=384\) |
| `superconductivity.npz` | UCI superconductivity (id 464) | \(n=21263\), \(p=81\) |
| `tecator.npz` | OpenML tecator (id 505) | \(n=240\), \(p\approx 124\) |
| `topo_2_1.npz` | OpenML topo_2_1 | \(n\approx 8885\), \(p=266\) |

Download automatically:

```bash
python examples/download_datasets.py
```

Loaders in `examples/data_loaders.py` read **only** these local files (no network at experiment time).
