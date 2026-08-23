"""Region-specific incremental feature menus for progressive feature revelation."""

from .conditional_greedy import (
    ConditionalGreedyResult,
    evaluate_conditional_greedy_progressive,
)
from .progressive import (
    ProgressiveCurve,
    compare_progressive_policies,
    evaluate_region_menus_progressive,
    format_curves_table,
)
from .region_menus import (
    AlternatingFitResult,
    RegionFeatureMenus,
    fit_region_menus,
    fit_region_menus_alternating,
    fit_region_menus_mixture_residual,
    mixture_gaussian_bic,
)
from .feature_catalog import (
    ct_feature_names,
    ct_pretty_name,
    superconductivity_feature_names,
    superconductivity_pretty_name,
)
from .regionalize import (
    mean_pairwise_prefix_jaccard,
    pick_by_inner_val,
    pick_by_menu_jaccard,
    pick_by_train_bic,
)
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
    "fit_region_menus_alternating",
    "fit_region_menus_mixture_residual",
    "mixture_gaussian_bic",
    "mean_pairwise_prefix_jaccard",
    "pick_by_inner_val",
    "pick_by_menu_jaccard",
    "pick_by_train_bic",
    "ct_feature_names",
    "ct_pretty_name",
    "superconductivity_feature_names",
    "superconductivity_pretty_name",
    "AlternatingFitResult",
    "ProgressiveCurve",
    "compare_progressive_policies",
    "evaluate_region_menus_progressive",
    "evaluate_conditional_greedy_progressive",
    "ConditionalGreedyResult",
    "format_curves_table",
]
