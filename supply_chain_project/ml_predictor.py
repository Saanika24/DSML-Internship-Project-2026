"""
ml_predictor.py
===============
Phase 2 Machine Learning Pipeline for AtmoGraph.

Builds strictly leakage-safe features from cleaned production datasets
(purchase_orders, shipments, supplier_performance_as_of_date, inventory,
demand_history, suppliers) combined with NetworkX graph structural & centrality
metrics from ``graph_utils.py`` and ``impact_analysis.py``.

Models trained & compared (chronological 80/20 train/test split):
  1. Purchase Order Delay Risk (Classification: ``is_delayed``)
  2. Purchase Order Delay Duration (Regression: ``days_late``)
  3. Shipment Delay Risk (Classification: ``is_delayed``)
  4. Shipment Delay Duration (Regression: ``days_late``)

Evaluates Random Forest, Gradient Boosting, and XGBoost, computes ROC-AUC,
PR-AUC, F1, Precision/Recall, Confusion Matrix, RMSE, MAE, R2, Feature
Importance, and SHAP summary values, saves trained bundles to ``models/``,
and feeds predicted risk probabilities into ``impact_analysis.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from data_pipeline import CleanedTables, load_cleaned_tables, load_full_supply_chain
from graph_utils import get_flow_graph, risk_level
from impact_analysis import (
    apply_ml_risk_weights,
    generate_ml_risk_disruptions,
    structural_features,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
PO_BUNDLE_PATH = MODELS_DIR / "po_model_bundle.joblib"
SHIPMENT_BUNDLE_PATH = MODELS_DIR / "shipment_model_bundle.joblib"
EVAL_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"

PO_NUMERIC_FEATURES = [
    "promised_lead_days",
    "quantity_ordered",
    "unit_cost",
    "total_cost",
    "order_month",
    "order_dayofweek",
    "order_quarter",
    "is_primary_supplier",
    "qty_to_weekly_demand_ratio",
    "qty_zscore_sku",
    "prior_completed_po_count",
    "prior_on_time_delivery_rate",
    "prior_average_days_late",
    "prior_late_order_count",
    "rolling_5_on_time_rate",
    "rolling_5_lead_time_std",
    "rolling_5_avg_days_late",
    "avg_lead_time_days",
    "lead_time_variability",
    "lead_time_gap",
    "payment_terms_days",
    "is_preferred",
    "annual_spend",
    "current_stock_qty",
    "safety_stock_qty",
    "reorder_point_qty",
    "days_of_stock",
    "avg_weekly_demand",
    "inventory_buffer_ratio",
    "stock_coverage_vs_lead_ratio",
    "is_below_safety_stock",
    "prior_week_demand",
    "sup_out_degree",
    "sup_downstream_count",
    "sup_pagerank",
    "sku_in_degree",
    "sku_max_dependency_in",
    "edge_dependency_score",
]

PO_CATEGORICAL_FEATURES = [
    "warehouse",
    "country",
    "category",
    "risk_tier",
    "abc_class",
]

SHIPMENT_NUMERIC_FEATURES = [
    "promised_transit_days",
    "quantity_shipped",
    "weight_kg",
    "shipping_cost",
    "cost_per_kg",
    "weight_per_unit",
    "ship_month_num",
    "ship_dayofweek",
    "carrier_prior_delay_rate",
    "carrier_prior_avg_days_late",
    "wh_prior_delay_rate",
    "current_stock_qty",
    "safety_stock_qty",
    "days_of_stock",
    "inventory_buffer_ratio",
    "sku_in_degree",
    "wh_in_degree",
    "wh_pagerank",
]

SHIPMENT_CATEGORICAL_FEATURES = [
    "carrier",
    "warehouse_origin",
    "category",
    "abc_class",
]


# ---------------------------------------------------------------------------
# Feature Engineering (Strictly Leakage-Safe)
# ---------------------------------------------------------------------------

def engineer_graph_features(graph: nx.DiGraph) -> pd.DataFrame:
    """Compute structural and centrality metrics for every entity in ``graph``."""
    struct = structural_features(graph)
    flow = get_flow_graph(graph)
    deg_cent = nx.degree_centrality(flow) if flow.number_of_nodes() > 1 else {}
    try:
        pagerank = nx.pagerank(flow, alpha=0.85) if flow.number_of_edges() > 0 else {}
    except Exception:
        pagerank = {n: 0.0 for n in flow.nodes}

    rows = []
    for node, feats in struct.items():
        rows.append(
            {
                **feats,
                "degree_centrality": round(float(deg_cent.get(node, 0.0)), 6),
                "pagerank": round(float(pagerank.get(node, 0.0)), 6),
            }
        )
    return pd.DataFrame(rows)


def _compute_causal_po_rolling_features(po: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly causal rolling supplier & SKU stats (no future leakage).

    For each purchase order placed on ``order_date``, only previously completed
    orders whose ``actual_delivery_date < order_date`` are included in the
    supplier's rolling-5 window, and only prior orders with ``order_date <
    current_order_date`` contribute to the SKU quantity anomaly z-score.
    """
    df = po.copy()
    df["_orig_idx"] = df.index
    df["_order_dt"] = pd.to_datetime(df["order_date"])
    df["_actual_dt"] = pd.to_datetime(df["actual_delivery_date"], errors="coerce")
    df["_promised_dt"] = pd.to_datetime(df["promised_delivery_date"])
    df["_actual_lead"] = (df["_actual_dt"] - df["_order_dt"]).dt.days
    df = df.sort_values(["_order_dt", "po_id"]).reset_index(drop=True)

    # 1. Causal rolling-5 completed orders per supplier (actual_delivery_date < current order_date)
    rolling_ot = np.full(len(df), np.nan)
    rolling_lt_std = np.full(len(df), np.nan)
    rolling_late_avg = np.full(len(df), np.nan)

    for _, grp in df.groupby("supplier_id", sort=False):
        idxs = grp.index.to_numpy()
        order_dts = grp["_order_dt"].to_numpy()
        actual_dts = grp["_actual_dt"].to_numpy()
        is_ot = grp["is_on_time"].astype(float).to_numpy()
        days_late = grp["days_late"].astype(float).to_numpy()
        actual_leads = grp["_actual_lead"].astype(float).to_numpy()

        # Sort completed events by actual_delivery_date so we advance pointer causally
        valid_mask = ~np.isnat(actual_dts)
        comp_order = np.argsort(actual_dts[valid_mask])
        comp_actual_dts = actual_dts[valid_mask][comp_order]
        comp_ot = is_ot[valid_mask][comp_order]
        comp_late = days_late[valid_mask][comp_order]
        comp_lead = actual_leads[valid_mask][comp_order]

        win_ot: deque = deque(maxlen=5)
        win_late: deque = deque(maxlen=5)
        win_lead: deque = deque(maxlen=5)
        ptr = 0
        n_comp = len(comp_actual_dts)

        for pos, row_idx in enumerate(idxs):
            cur_dt = order_dts[pos]
            while ptr < n_comp and comp_actual_dts[ptr] < cur_dt:
                win_ot.append(comp_ot[ptr])
                win_late.append(comp_late[ptr])
                win_lead.append(comp_lead[ptr])
                ptr += 1
            if win_ot:
                rolling_ot[row_idx] = float(np.mean(win_ot))
                rolling_late_avg[row_idx] = float(np.mean(win_late))
                rolling_lt_std[row_idx] = float(np.std(win_lead)) if len(win_lead) > 1 else 0.0

    # 2. Causal SKU order quantity anomaly z-score (expanding prior orders for same SKU)
    qty_z = np.zeros(len(df), dtype=float)
    for _, grp in df.groupby("sku_id", sort=False):
        idxs = grp.index.to_numpy()
        qtys = grp["quantity_ordered"].astype(float).to_numpy()
        prior_qtys: List[float] = []
        cur_day = None
        day_buffer: List[float] = []
        order_dts = grp["_order_dt"].to_numpy()

        for pos, row_idx in enumerate(idxs):
            dt = order_dts[pos]
            if cur_day is None or dt > cur_day:
                if day_buffer:
                    prior_qtys.extend(day_buffer)
                    day_buffer = []
                cur_day = dt
            if len(prior_qtys) >= 2:
                mean_q = float(np.mean(prior_qtys))
                std_q = float(np.std(prior_qtys))
                if std_q > 1e-6:
                    qty_z[row_idx] = (qtys[pos] - mean_q) / std_q
            day_buffer.append(qtys[pos])

    df["rolling_5_on_time_rate"] = rolling_ot
    df["rolling_5_lead_time_std"] = rolling_lt_std
    df["rolling_5_avg_days_late"] = rolling_late_avg
    df["qty_zscore_sku"] = np.clip(qty_z, -5.0, 5.0)
    return df.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)


def build_po_feature_frame(
    tables: CleanedTables,
    graph: nx.DiGraph,
    po_override: Optional[pd.DataFrame] = None,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Build leakage-safe order-time feature DataFrame for Purchase Orders."""
    po_src = tables.purchase_orders if po_override is None else po_override
    if po_override is None:
        df = _compute_causal_po_rolling_features(po_src)
    else:
        # For live scenario scoring, attach latest known supplier stats safely
        df = po_src.copy()
        df["_order_dt"] = pd.to_datetime(df["order_date"])
        df["_promised_dt"] = pd.to_datetime(df["promised_delivery_date"])

        hist_completed = tables.purchase_orders.copy()
        hist_completed["_actual_dt"] = pd.to_datetime(hist_completed["actual_delivery_date"], errors="coerce")
        hist_completed["_order_dt"] = pd.to_datetime(hist_completed["order_date"])
        hist_completed["_lead"] = (hist_completed["_actual_dt"] - hist_completed["_order_dt"]).dt.days
        hist_completed = hist_completed.dropna(subset=["_actual_dt"]).sort_values("_actual_dt")

        last5 = hist_completed.groupby("supplier_id").tail(5).groupby("supplier_id").agg(
            rolling_5_on_time_rate=("is_on_time", "mean"),
            rolling_5_lead_time_std=("_lead", "std"),
            rolling_5_avg_days_late=("days_late", "mean"),
        ).fillna(0.0)
        sku_stats = tables.purchase_orders.groupby("sku_id")["quantity_ordered"].agg(["mean", "std"]).fillna(1.0)
        df = df.merge(last5, on="supplier_id", how="left")
        df = df.merge(sku_stats, on="sku_id", how="left")
        df["qty_zscore_sku"] = np.where(
            df["std"] > 1e-6,
            np.clip((df["quantity_ordered"].astype(float) - df["mean"]) / df["std"], -5.0, 5.0),
            0.0,
        )
        df = df.drop(columns=["mean", "std"])

    # Join leakage-safe as-of-date supplier snapshot
    perf = tables.supplier_performance_as_of_date.drop_duplicates(["supplier_id", "as_of_date"])
    if po_override is None:
        df = df.merge(
            perf,
            left_on=["supplier_id", "order_date"],
            right_on=["supplier_id", "as_of_date"],
            how="left",
        )
    else:
        latest_perf = (
            perf.sort_values("as_of_date").groupby("supplier_id").tail(1).drop(columns=["as_of_date"])
        )
        df = df.merge(latest_perf, on="supplier_id", how="left")

    # Join static supplier attributes (excluding full-period on_time_delivery_rate/quality_acceptance_rate)
    sup_cols = [
        "supplier_id",
        "country",
        "category",
        "avg_lead_time_days",
        "lead_time_variability",
        "payment_terms_days",
        "is_preferred",
        "risk_tier",
        "annual_spend",
    ]
    df = df.merge(tables.suppliers[sup_cols], on="supplier_id", how="left")

    # Join inventory buffer attributes
    inv_cols = [
        "sku_id",
        "primary_supplier_id",
        "current_stock_qty",
        "safety_stock_qty",
        "reorder_point_qty",
        "days_of_stock",
        "avg_weekly_demand",
        "abc_class",
    ]
    df = df.merge(tables.inventory[inv_cols], on="sku_id", how="left")

    # Join latest prior weekly demand strictly before order_date via merge_asof
    dem = tables.demand_history[["sku_id", "warehouse", "week_start", "demand_units"]].copy()
    dem["_dem_dt"] = pd.to_datetime(dem["week_start"])
    dem = dem.sort_values("_dem_dt")
    df["_orig_row"] = np.arange(len(df))
    df = df.sort_values("_order_dt")
    df = pd.merge_asof(
        df,
        dem[["sku_id", "warehouse", "_dem_dt", "demand_units"]].rename(columns={"demand_units": "prior_week_demand"}),
        left_on="_order_dt",
        right_on="_dem_dt",
        by=["sku_id", "warehouse"],
        direction="backward",
        allow_exact_matches=False,
    )
    df = df.sort_values("_orig_row").drop(columns=["_orig_row", "_dem_dt"], errors="ignore").reset_index(drop=True)

    # Calendar & engineered ratios
    df["promised_lead_days"] = (df["_promised_dt"] - df["_order_dt"]).dt.days.astype(float)
    df["order_month"] = df["_order_dt"].dt.month.astype(float)
    df["order_dayofweek"] = df["_order_dt"].dt.day_of_week.astype(float)
    df["order_quarter"] = df["_order_dt"].dt.quarter.astype(float)
    df["is_preferred"] = df["is_preferred"].fillna(False).astype(int)
    df["is_primary_supplier"] = (df["supplier_id"] == df["primary_supplier_id"]).astype(int)
    df["lead_time_gap"] = df["promised_lead_days"] - df["avg_lead_time_days"].astype(float)
    df["qty_to_weekly_demand_ratio"] = df["quantity_ordered"].astype(float) / np.maximum(
        1.0, df["avg_weekly_demand"].astype(float)
    )
    df["inventory_buffer_ratio"] = df["current_stock_qty"].astype(float) / np.maximum(
        1.0, df["safety_stock_qty"].astype(float)
    )
    df["stock_coverage_vs_lead_ratio"] = df["days_of_stock"].astype(float) / np.maximum(
        1.0, df["promised_lead_days"]
    )
    df["is_below_safety_stock"] = (
        df["current_stock_qty"].astype(float) < df["safety_stock_qty"].astype(float)
    ).astype(int)

    # Graph structural & centrality features
    g_feats = engineer_graph_features(graph).set_index("entity_id")
    df["sup_out_degree"] = df["supplier_id"].map(g_feats["out_degree"]).fillna(0.0)
    df["sup_downstream_count"] = df["supplier_id"].map(g_feats["downstream_count"]).fillna(0.0)
    df["sup_pagerank"] = df["supplier_id"].map(g_feats["pagerank"]).fillna(0.0)
    df["sku_in_degree"] = df["sku_id"].map(g_feats["in_degree"]).fillna(0.0)
    df["sku_max_dependency_in"] = df["sku_id"].map(g_feats["max_dependency_in"]).fillna(0.0)
    edge_dep = {(u, v): d.get("dependency_score", 0.0) for u, v, d in graph.edges(data=True)}
    df["edge_dependency_score"] = [
        float(edge_dep.get((s, k), 0.0)) for s, k in zip(df["supplier_id"], df["sku_id"])
    ]

    if include_targets and "is_on_time" in df.columns:
        df["is_delayed"] = (~df["is_on_time"].astype(bool)).astype(int)
        df["days_late"] = df["days_late"].astype(float)

    return df


def _compute_causal_group_rates(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    actual_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Compute strictly causal prior delay rate & mean days late per group."""
    work = df[[group_col, date_col, actual_col, "is_on_time", "days_late"]].copy()
    work["_orig"] = np.arange(len(work))
    work["_dt"] = pd.to_datetime(work[date_col])
    work["_act"] = pd.to_datetime(work[actual_col], errors="coerce")
    work = work.sort_values(["_dt", "_orig"]).reset_index(drop=True)

    rate_out = np.full(len(work), np.nan)
    late_out = np.full(len(work), np.nan)

    for _, grp in work.groupby(group_col, sort=False):
        idxs = grp.index.to_numpy()
        dts = grp["_dt"].to_numpy()
        acts = grp["_act"].to_numpy()
        delayed = (~grp["is_on_time"].astype(bool)).astype(float).to_numpy()
        dl = grp["days_late"].astype(float).to_numpy()

        valid = ~np.isnat(acts)
        order = np.argsort(acts[valid])
        c_acts = acts[valid][order]
        c_del = delayed[valid][order]
        c_dl = dl[valid][order]

        ptr = 0
        count = 0
        sum_del = 0.0
        sum_dl = 0.0
        for pos, row_idx in enumerate(idxs):
            cur = dts[pos]
            while ptr < len(c_acts) and c_acts[ptr] < cur:
                count += 1
                sum_del += c_del[ptr]
                sum_dl += c_dl[ptr]
                ptr += 1
            if count > 0:
                rate_out[row_idx] = sum_del / count
                late_out[row_idx] = sum_dl / count

    work[f"{prefix}_prior_delay_rate"] = rate_out
    work[f"{prefix}_prior_avg_days_late"] = late_out
    return work.sort_values("_orig").reset_index(drop=True)[[f"{prefix}_prior_delay_rate", f"{prefix}_prior_avg_days_late"]]


def build_shipment_feature_frame(
    tables: CleanedTables,
    graph: nx.DiGraph,
    ship_override: Optional[pd.DataFrame] = None,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Build leakage-safe dispatch-time feature DataFrame for Shipments."""
    ship_src = tables.shipments if ship_override is None else ship_override
    df = ship_src.copy()
    df["_ship_dt"] = pd.to_datetime(df["ship_date"])
    df["_promised_dt"] = pd.to_datetime(df["promised_delivery_date"])

    if ship_override is None:
        carrier_stats = _compute_causal_group_rates(
            df, "carrier", "ship_date", "actual_delivery_date", "carrier"
        )
        wh_stats = _compute_causal_group_rates(
            df, "warehouse_origin", "ship_date", "actual_delivery_date", "wh"
        )
        df["carrier_prior_delay_rate"] = carrier_stats["carrier_prior_delay_rate"]
        df["carrier_prior_avg_days_late"] = carrier_stats["carrier_prior_avg_days_late"]
        df["wh_prior_delay_rate"] = wh_stats["wh_prior_delay_rate"]
    else:
        hist = tables.shipments.copy()
        hist["_del"] = (~hist["is_on_time"].astype(bool)).astype(float)
        c_agg = hist.groupby("carrier").agg(
            carrier_prior_delay_rate=("_del", "mean"),
            carrier_prior_avg_days_late=("days_late", "mean"),
        )
        w_agg = hist.groupby("warehouse_origin").agg(wh_prior_delay_rate=("_del", "mean"))
        df = df.merge(c_agg, on="carrier", how="left")
        df = df.merge(w_agg, on="warehouse_origin", how="left")

    inv_cols = [
        "sku_id",
        "category",
        "abc_class",
        "current_stock_qty",
        "safety_stock_qty",
        "days_of_stock",
    ]
    df = df.merge(tables.inventory[inv_cols], on="sku_id", how="left")

    df["promised_transit_days"] = (df["_promised_dt"] - df["_ship_dt"]).dt.days.astype(float)
    df["ship_month_num"] = df["_ship_dt"].dt.month.astype(float)
    df["ship_dayofweek"] = df["_ship_dt"].dt.day_of_week.astype(float)
    df["cost_per_kg"] = df["shipping_cost"].astype(float) / np.maximum(0.1, df["weight_kg"].astype(float))
    df["weight_per_unit"] = df["weight_kg"].astype(float) / np.maximum(1.0, df["quantity_shipped"].astype(float))
    df["inventory_buffer_ratio"] = df["current_stock_qty"].astype(float) / np.maximum(
        1.0, df["safety_stock_qty"].astype(float)
    )

    g_feats = engineer_graph_features(graph).set_index("entity_id")
    df["sku_in_degree"] = df["sku_id"].map(g_feats["in_degree"]).fillna(0.0)
    df["wh_in_degree"] = df["warehouse_origin"].map(g_feats["in_degree"]).fillna(0.0)
    df["wh_pagerank"] = df["warehouse_origin"].map(g_feats["pagerank"]).fillna(0.0)

    if include_targets and "is_on_time" in df.columns:
        df["is_delayed"] = (~df["is_on_time"].astype(bool)).astype(int)
        df["days_late"] = df["days_late"].astype(float)

    return df


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def _classification_candidates(pos_weight: float) -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=140,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=25,
            reg_alpha=2.0,
            reg_lambda=20.0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def _regression_candidates() -> Dict[str, object]:
    return {
        "Ridge": Ridge(alpha=100.0, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            max_depth=7,
            min_samples_leaf=25,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=140,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=25,
            reg_alpha=2.0,
            reg_lambda=25.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


def _extract_feature_importance(pipe: Pipeline, num_cols: List[str], cat_cols: List[str]) -> List[Dict]:
    prep: ColumnTransformer = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    cat_names = list(prep.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols))
    feat_names = list(num_cols) + cat_names
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        coef = getattr(model, "coef_", None)
        if coef is None:
            return []
        raw = np.abs(np.asarray(coef).ravel())
        total = float(raw.sum()) or 1.0
        importances = raw / total
    pairs = [
        {"feature": name, "importance": round(float(val), 5)}
        for name, val in zip(feat_names, importances)
    ]
    pairs.sort(key=lambda x: -x["importance"])
    return pairs[:25]


def _extract_shap_summary(pipe: Pipeline, X_sample: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> List[Dict]:
    """Compute mean |SHAP| values for the winning model on a test sample."""
    try:
        import shap

        prep: ColumnTransformer = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        X_trans = prep.transform(X_sample)
        cat_names = list(prep.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols))
        feat_names = list(num_cols) + cat_names
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_trans)
            shap_vals = explainer.shap_values(X_trans)
        else:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_trans, check_additivity=False)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        if getattr(shap_vals, "ndim", 2) == 3:
            shap_vals = shap_vals[:, :, 1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        summary = [
            {"feature": name, "mean_abs_shap": round(float(val), 5)}
            for name, val in zip(feat_names, mean_abs)
        ]
        summary.sort(key=lambda x: -x["mean_abs_shap"])
        return summary[:20]
    except Exception:
        return []


def _train_domain_suite(
    df: pd.DataFrame,
    date_col: str,
    num_cols: List[str],
    cat_cols: List[str],
    domain_name: str,
    train_frac: float = 0.80,
) -> Tuple[Dict, Dict]:
    """Chronologically split ``df`` and train classification + regression suites."""
    ordered = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(ordered) * train_frac)
    train_df = ordered.iloc[:split_idx].reset_index(drop=True)
    test_df = ordered.iloc[split_idx:].reset_index(drop=True)

    feature_cols = num_cols + cat_cols
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train_cls, y_test_cls = train_df["is_delayed"].to_numpy(), test_df["is_delayed"].to_numpy()
    y_train_reg, y_test_reg = train_df["days_late"].to_numpy(), test_df["days_late"].to_numpy()

    pos_count = max(1, int(y_train_cls.sum()))
    neg_count = max(1, int(len(y_train_cls) - pos_count))
    pos_weight = float(neg_count / pos_count)

    # 1. Classification comparison
    cls_metrics: Dict[str, Dict] = {}
    cls_pipelines: Dict[str, Pipeline] = {}
    bal_weights = compute_sample_weight("balanced", y_train_cls)
    for name, estimator in _classification_candidates(pos_weight).items():
        print(f"Training {domain_name} classifier: {name}...", flush=True)
        pipe = Pipeline([("prep", _make_preprocessor(num_cols, cat_cols)), ("model", estimator)])
        if name == "GradientBoosting":
            pipe.fit(X_train, y_train_cls, model__sample_weight=bal_weights)
        else:
            pipe.fit(X_train, y_train_cls)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Also track best F1 threshold for imbalanced inspection
        fpr, tpr, thresholds = roc_curve(y_test_cls, probs)
        step = max(1, len(fpr) // 60)
        cm = confusion_matrix(y_test_cls, preds).tolist()
        cls_metrics[name] = {
            "roc_auc": round(float(roc_auc_score(y_test_cls, probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test_cls, probs)), 4),
            "f1": round(float(f1_score(y_test_cls, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test_cls, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test_cls, preds, zero_division=0)), 4),
            "accuracy": round(float((preds == y_test_cls).mean()), 4),
            "confusion_matrix": cm,
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr[::step]],
                "tpr": [round(float(x), 4) for x in tpr[::step]],
            },
            "feature_importance": _extract_feature_importance(pipe, num_cols, cat_cols),
        }
        cls_pipelines[name] = pipe

    best_cls_name = max(cls_metrics, key=lambda k: cls_metrics[k]["roc_auc"])
    best_cls_pipe = cls_pipelines[best_cls_name]

    # 2. Regression comparison (days_late)
    reg_metrics: Dict[str, Dict] = {}
    reg_pipelines: Dict[str, Pipeline] = {}
    for name, estimator in _regression_candidates().items():
        print(f"Training {domain_name} regressor: {name}...", flush=True)
        pipe = Pipeline([("prep", _make_preprocessor(num_cols, cat_cols)), ("model", estimator)])
        pipe.fit(X_train, y_train_reg)
        preds = np.clip(pipe.predict(X_test), 0.0, None)
        reg_metrics[name] = {
            "rmse": round(float(root_mean_squared_error(y_test_reg, preds)), 4),
            "mae": round(float(mean_absolute_error(y_test_reg, preds)), 4),
            "r2": round(float(r2_score(y_test_reg, preds)), 4),
            "feature_importance": _extract_feature_importance(pipe, num_cols, cat_cols),
        }
        reg_pipelines[name] = pipe

    best_reg_name = min(reg_metrics, key=lambda k: reg_metrics[k]["rmse"])
    best_reg_pipe = reg_pipelines[best_reg_name]

    # SHAP summary for the winning classifier on up to 150 test records
    print(f"Computing SHAP summary for {domain_name} ({best_cls_name})...", flush=True)
    sample_X = X_test.iloc[: min(150, len(X_test))]
    shap_summary = _extract_shap_summary(best_cls_pipe, sample_X, num_cols, cat_cols)

    bundle = {
        "domain": domain_name,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_classifier_name": best_cls_name,
        "classifier": best_cls_pipe,
        "best_regressor_name": best_reg_name,
        "regressor": best_reg_pipe,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": [str(train_df[date_col].min()), str(train_df[date_col].max())],
        "test_date_range": [str(test_df[date_col].min()), str(test_df[date_col].max())],
    }

    eval_report = {
        "domain": domain_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_positive_rate": round(float(y_test_cls.mean()), 4),
        "test_mean_days_late": round(float(y_test_reg.mean()), 4),
        "train_date_range": bundle["train_date_range"],
        "test_date_range": bundle["test_date_range"],
        "best_classifier": best_cls_name,
        "best_regressor": best_reg_name,
        "classification_models": cls_metrics,
        "regression_models": reg_metrics,
        "shap_summary": shap_summary,
    }
    return bundle, eval_report


def train_and_save_models(
    data_dir: Optional[str | Path] = None,
    models_dir: str | Path = MODELS_DIR,
) -> Dict:
    """End-to-end training of PO and Shipment models + evaluation artifact export."""
    tables = load_cleaned_tables(data_dir)
    sc = load_full_supply_chain(data_dir=tables.data_dir, strict=True)

    po_df = build_po_feature_frame(tables, sc.graph, include_targets=True)
    ship_df = build_shipment_feature_frame(tables, sc.graph, include_targets=True)

    po_bundle, po_eval = _train_domain_suite(
        po_df,
        date_col="order_date",
        num_cols=PO_NUMERIC_FEATURES,
        cat_cols=PO_CATEGORICAL_FEATURES,
        domain_name="purchase_orders",
    )
    ship_bundle, ship_eval = _train_domain_suite(
        ship_df,
        date_col="ship_date",
        num_cols=SHIPMENT_NUMERIC_FEATURES,
        cat_cols=SHIPMENT_CATEGORICAL_FEATURES,
        domain_name="shipments",
    )

    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(po_bundle, out_dir / "po_model_bundle.joblib")
    joblib.dump(ship_bundle, out_dir / "shipment_model_bundle.joblib")

    evaluation = {
        "purchase_orders": po_eval,
        "shipments": ship_eval,
    }
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    return evaluation


def load_model_bundles(models_dir: str | Path = MODELS_DIR) -> Tuple[Dict, Dict, Dict]:
    """Load saved PO bundle, Shipment bundle, and evaluation metrics JSON."""
    out_dir = Path(models_dir)
    po_bundle = joblib.load(out_dir / "po_model_bundle.joblib")
    ship_bundle = joblib.load(out_dir / "shipment_model_bundle.joblib")
    with open(out_dir / "evaluation_results.json", "r", encoding="utf-8") as f:
        evaluation = json.load(f)
    return po_bundle, ship_bundle, evaluation


# ---------------------------------------------------------------------------
# Inference & Dynamic Graph Risk Scoring
# ---------------------------------------------------------------------------

def _mitigation_action(prob: float, expected_days: float, is_below_safety: int) -> str:
    if prob >= 0.65 or expected_days >= 5.0:
        if is_below_safety:
            return "URGENT: Expedite air-freight & split order to secondary supplier (below safety stock)"
        return "HIGH RISK: Activate backup supplier allocation & advance reorder point"
    if prob >= 0.40 or expected_days >= 2.0:
        return "MODERATE: Increase safety buffer & confirm carrier SLA window"
    return "LOW: Standard monitoring"


def score_purchase_orders(
    po_bundle: Dict,
    tables: CleanedTables,
    graph: nx.DiGraph,
    po_subset: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score Purchase Orders for delay probability and expected delay duration."""
    if po_subset is None:
        feat_df = build_po_feature_frame(tables, graph, include_targets=True)
    else:
        feat_df = build_po_feature_frame(tables, graph, po_override=po_subset, include_targets=False)

    X = feat_df[po_bundle["feature_cols"]]
    probs = po_bundle["classifier"].predict_proba(X)[:, 1]
    exp_days = np.clip(po_bundle["regressor"].predict(X), 0.0, None)

    out = feat_df[
        [
            "po_id",
            "supplier_id",
            "sku_id",
            "warehouse",
            "order_date",
            "promised_delivery_date",
            "quantity_ordered",
            "total_cost",
            "po_status",
            "promised_lead_days",
            "prior_on_time_delivery_rate",
            "days_of_stock",
            "is_below_safety_stock",
            "edge_dependency_score",
        ]
    ].copy()
    out["delay_probability"] = np.round(probs, 4)
    out["expected_days_late"] = np.round(exp_days, 2)
    out["predicted_severity_score"] = np.round(np.clip(exp_days / 14.0, 0.0, 1.0), 4)
    out["risk_level"] = [risk_level(p) for p in out["delay_probability"]]
    out["recommended_action"] = [
        _mitigation_action(p, d, s)
        for p, d, s in zip(out["delay_probability"], out["expected_days_late"], out["is_below_safety_stock"])
    ]
    return out.sort_values("delay_probability", ascending=False).reset_index(drop=True)


def score_shipments(
    ship_bundle: Dict,
    tables: CleanedTables,
    graph: nx.DiGraph,
    ship_subset: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score Shipments for transit delay probability and expected days late."""
    if ship_subset is None:
        feat_df = build_shipment_feature_frame(tables, graph, include_targets=True)
    else:
        feat_df = build_shipment_feature_frame(tables, graph, ship_override=ship_subset, include_targets=False)

    X = feat_df[ship_bundle["feature_cols"]]
    probs = ship_bundle["classifier"].predict_proba(X)[:, 1]
    exp_days = np.clip(ship_bundle["regressor"].predict(X), 0.0, None)

    out = feat_df[
        [
            "shipment_id",
            "sku_id",
            "warehouse_origin",
            "carrier",
            "ship_date",
            "promised_delivery_date",
            "quantity_shipped",
            "shipping_cost",
            "promised_transit_days",
            "days_of_stock",
        ]
    ].copy()
    out["delay_probability"] = np.round(probs, 4)
    out["expected_days_late"] = np.round(exp_days, 2)
    out["risk_level"] = [risk_level(p) for p in out["delay_probability"]]
    return out.sort_values("delay_probability", ascending=False).reset_index(drop=True)


def compute_dynamic_graph_risks(
    po_scored: pd.DataFrame,
    graph: nx.DiGraph,
    risk_blend: float = 0.5,
) -> Tuple[nx.DiGraph, Dict[str, float], Dict[str, float], List[Dict]]:
    """Aggregate ML predictions onto nodes/edges and return a risk-weighted graph.

    Returns (weighted_graph, node_risks, expected_delays, ml_disruptions).
    """
    # Use recent PO predictions (last 90 days of dataset) to reflect current supplier risk
    latest_date = pd.to_datetime(po_scored["order_date"]).max()
    recent = po_scored[pd.to_datetime(po_scored["order_date"]) >= (latest_date - pd.Timedelta(days=90))]
    if recent.empty:
        recent = po_scored

    sup_agg = recent.groupby("supplier_id").agg(
        prob=("delay_probability", "mean"),
        delay=("expected_days_late", "mean"),
    )
    sku_agg = recent.groupby("sku_id").agg(
        prob=("delay_probability", "mean"),
        delay=("expected_days_late", "mean"),
    )
    edge_agg = recent.groupby(["supplier_id", "sku_id"])["delay_probability"].mean().to_dict()

    node_risks: Dict[str, float] = {}
    expected_delays: Dict[str, float] = {}
    for sid, row in sup_agg.iterrows():
        node_risks[sid] = float(row["prob"])
        expected_delays[sid] = float(row["delay"])
    for kid, row in sku_agg.iterrows():
        node_risks[kid] = float(row["prob"])
        expected_delays[kid] = float(row["delay"])

    weighted_graph = apply_ml_risk_weights(
        graph, node_risks=node_risks, edge_risks=edge_agg, risk_blend=risk_blend
    )
    ml_disruptions = generate_ml_risk_disruptions(
        weighted_graph,
        node_risks=node_risks,
        threshold=0.40,
        expected_delays=expected_delays,
        entity_types=("supplier",),
        as_of_date=str(latest_date.date()),
    )
    return weighted_graph, node_risks, expected_delays, ml_disruptions


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate AtmoGraph ML models.")
    parser.add_argument("--data-dir", default=None, help="Directory with cleaned CSVs")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Where to save model artifacts")
    args = parser.parse_args()

    eval_results = train_and_save_models(data_dir=args.data_dir, models_dir=args.models_dir)
    for domain in ("purchase_orders", "shipments"):
        d = eval_results[domain]
        print(f"\n=== {domain.upper()} (train={d['train_rows']}, test={d['test_rows']}) ===")
        print(f"Best Classifier: {d['best_classifier']} | Metrics:", d["classification_models"][d["best_classifier"]]["roc_auc"], "ROC-AUC")
        for mname, mvals in d["classification_models"].items():
            print(f"  [CLS] {mname:18s} ROC-AUC={mvals['roc_auc']:.4f} PR-AUC={mvals['pr_auc']:.4f} F1={mvals['f1']:.4f} Prec={mvals['precision']:.4f} Rec={mvals['recall']:.4f}")
        print(f"Best Regressor: {d['best_regressor']}")
        for mname, mvals in d["regression_models"].items():
            print(f"  [REG] {mname:18s} RMSE={mvals['rmse']:.4f} MAE={mvals['mae']:.4f} R2={mvals['r2']:.4f}")
        print("  Top 5 SHAP features:", d["shap_summary"][:5])


if __name__ == "__main__":
    main()
