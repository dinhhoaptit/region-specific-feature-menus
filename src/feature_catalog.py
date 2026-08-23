"""UCI column names for the paper's public regression datasets."""

from __future__ import annotations

_SC_PROPERTIES = (
    "atomic_mass",
    "fie",
    "atomic_radius",
    "Density",
    "ElectronAffinity",
    "FusionHeat",
    "ThermalConductivity",
    "Valence",
)
_SC_STATS = (
    "mean",
    "wtd_mean",
    "gmean",
    "wtd_gmean",
    "entropy",
    "wtd_entropy",
    "range",
    "wtd_range",
    "std",
    "wtd_std",
)

_STAT_PRETTY = {
    "mean": "mean",
    "wtd_mean": "weighted mean",
    "gmean": "geometric mean",
    "wtd_gmean": "weighted geometric mean",
    "entropy": "entropy",
    "wtd_entropy": "weighted entropy",
    "range": "range",
    "wtd_range": "weighted range",
    "std": "standard deviation",
    "wtd_std": "weighted standard deviation",
}
_PROP_PRETTY = {
    "atomic_mass": "atomic mass",
    "fie": "first ionization energy",
    "atomic_radius": "atomic radius",
    "Density": "density",
    "ElectronAffinity": "electron affinity",
    "FusionHeat": "fusion heat",
    "ThermalConductivity": "thermal conductivity",
    "Valence": "valence",
}


def superconductivity_feature_names() -> list[str]:
    """Hamidieh (2018) / UCI 464 ``train.csv`` column order, excluding the target."""
    names = ["number_of_elements"]
    for prop in _SC_PROPERTIES:
        for stat in _SC_STATS:
            names.append(f"{stat}_{prop}")
    return names


def superconductivity_pretty_name(index: int) -> str:
    raw = superconductivity_feature_names()[int(index)]
    if raw == "number_of_elements":
        return "number of elements"
    for s in sorted(_SC_STATS, key=len, reverse=True):
        prefix = s + "_"
        if raw.startswith(prefix):
            prop = raw[len(prefix) :]
            return f"{_STAT_PRETTY[s]} {_PROP_PRETTY[prop]}"
    return raw.replace("_", " ")


def ct_feature_names() -> list[str]:
    """UCI 206: patient id dropped; 240 bone polar bins then 144 air polar bins."""
    names = [f"bone_polar_bin_{i}" for i in range(240)]
    names.extend(f"air_polar_bin_{i}" for i in range(144))
    return names


def ct_pretty_name(index: int) -> str:
    j = int(index)
    if j < 240:
        return f"bone polar bin {j}"
    return f"air polar bin {j - 240}"
