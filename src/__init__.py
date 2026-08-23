"""Region-specific incremental feature menus for progressive feature revelation."""

from .progressive import (
    ProgressiveCurve,
    compare_progressive_policies,
    evaluate_region_menus_progressive,
    format_curves_table,
)
from .region_menus import RegionFeatureMenus, fit_region_menus
from .selectors import SelectionResult, select_features, select_lasso_lars, select_forward_stepwise
from .vif import VIFResult, vif_regression

__all__ = [
    "VIFResult",
    "vif_regression",
    "SelectionResult",
    "select_features",
    "select_lasso_lars",
    "select_forward_stepwise",
    "RegionFeatureMenus",
    "fit_region_menus",
    "ProgressiveCurve",
    "compare_progressive_policies",
    "evaluate_region_menus_progressive",
    "format_curves_table",
]
