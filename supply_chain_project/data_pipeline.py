"""
data_pipeline.py
================
Phase 1 Data Engineering & Graph Ingestion Adapter for AtmoGraph.

Unifies the cleaned multi-table CSV dataset (suppliers, inventory, purchase_orders,
shipments, demand_history, supplier_performance_as_of_date, and
supplier_sku_warehouse_relationships) into the validated node/edge schema
expected by ``graph_builder.py`` and ``impact_analysis.py``.

Main entry points
-----------------
resolve_data_dir(data_dir=None)
load_cleaned_tables(data_dir=None)
build_enriched_tables_from_cleaned(tables, ...) -> (entities_df, relationships_df, report)
load_full_supply_chain(data_dir=None, disruptions_csv=None, ...) -> SupplyChainData
adapt_disruption_events_df(events_df) -> pd.DataFrame
compute_executive_kpis(tables, graph) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import networkx as nx
import pandas as pd

from graph_builder import (
    REQUIRED_DISRUPTION_COLUMNS,
    REQUIRED_ENTITY_COLUMNS,
    REQUIRED_RELATIONSHIP_COLUMNS,
    SupplyChainData,
    build_supply_chain,
    load_csv,
    purchase_orders_to_tables,
)
from graph_utils import ValidationReport


DEFAULT_DATA_FILENAMES = (
    "suppliers.csv",
    "inventory.csv",
    "purchase_orders.csv",
    "shipments.csv",
    "demand_history.csv",
    "supplier_performance_as_of_date.csv",
    "supplier_historical_performance.csv",
    "supplier_sku_warehouse_relationships.csv",
)

# Column mapping from cleaned purchase_orders.csv to graph_builder's PO schema
CLEANED_PO_COLUMN_MAP: Dict[str, str] = {
    "supplier_id": "supplier_id",
    "sku_id": "sku_id",
    "warehouse_id": "warehouse",
    "quantity": "quantity_ordered",
}


@dataclass
class CleanedTables:
    """Container for all cleaned production CSV tables."""

    data_dir: Path
    suppliers: pd.DataFrame
    inventory: pd.DataFrame
    purchase_orders: pd.DataFrame
    shipments: pd.DataFrame
    demand_history: pd.DataFrame
    supplier_performance_as_of_date: pd.DataFrame
    supplier_historical_performance: pd.DataFrame
    relationships_evidence: pd.DataFrame


def resolve_data_dir(data_dir: Optional[str | Path] = None) -> Path:
    """Locate the directory containing the cleaned production CSV files.

    Checks ``data_dir`` if provided; otherwise checks the workspace root
    (``../`` relative to ``supply_chain_project/``) and
    ``../supply-chain-cleaned-for-sharing/``.
    """
    if data_dir is not None:
        candidate = Path(data_dir).resolve()
        if (candidate / "purchase_orders.csv").is_file():
            return candidate
        raise FileNotFoundError(f"Cleaned CSVs not found in {candidate}")

    project_dir = Path(__file__).resolve().parent
    candidates = [
        project_dir.parent,
        project_dir.parent / "supply-chain-cleaned-for-sharing",
        project_dir / "supply-chain-cleaned-for-sharing",
    ]
    for candidate in candidates:
        if (candidate / "purchase_orders.csv").is_file() and (candidate / "suppliers.csv").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate cleaned supply chain CSVs (suppliers.csv, purchase_orders.csv, ...)."
    )


def load_cleaned_tables(data_dir: Optional[str | Path] = None) -> CleanedTables:
    """Load all cleaned production CSV tables with standardized types."""
    root = resolve_data_dir(data_dir)

    suppliers = pd.read_csv(root / "suppliers.csv")
    inventory = pd.read_csv(root / "inventory.csv")
    purchase_orders = pd.read_csv(root / "purchase_orders.csv")
    shipments = pd.read_csv(root / "shipments.csv")
    demand_history = pd.read_csv(root / "demand_history.csv")
    perf_as_of = pd.read_csv(root / "supplier_performance_as_of_date.csv")
    hist_perf = pd.read_csv(root / "supplier_historical_performance.csv")
    rel_evidence = pd.read_csv(root / "supplier_sku_warehouse_relationships.csv")

    return CleanedTables(
        data_dir=root,
        suppliers=suppliers,
        inventory=inventory,
        purchase_orders=purchase_orders,
        shipments=shipments,
        demand_history=demand_history,
        supplier_performance_as_of_date=perf_as_of,
        supplier_historical_performance=hist_perf,
        relationships_evidence=rel_evidence,
    )


def _enrich_entities(
    base_entities_df: pd.DataFrame,
    tables: CleanedTables,
) -> pd.DataFrame:
    """Attach rich supplier, SKU, and warehouse attributes to graph_builder entities."""
    suppliers_idx = tables.suppliers.drop_duplicates("supplier_id").set_index("supplier_id")
    inventory_idx = tables.inventory.drop_duplicates("sku_id").set_index("sku_id")

    po_wh_vol = (
        tables.purchase_orders.groupby("warehouse")["quantity_ordered"]
        .sum()
        .to_dict()
    )
    inv_wh_stats = (
        tables.inventory.groupby("warehouse")
        .agg(
            sku_count=("sku_id", "nunique"),
            total_stock_qty=("current_stock_qty", "sum"),
            below_safety_count=(
                "current_stock_qty",
                lambda s: int((s < tables.inventory.loc[s.index, "safety_stock_qty"]).sum()),
            ),
        )
        .to_dict(orient="index")
    )

    extra_cols = [
        "category",
        "risk_tier",
        "is_preferred",
        "avg_lead_time_days",
        "lead_time_variability",
        "annual_spend",
        "on_time_delivery_rate",
        "quality_acceptance_rate",
        "primary_supplier_id",
        "home_warehouse",
        "current_stock_qty",
        "safety_stock_qty",
        "reorder_point_qty",
        "max_stock_qty",
        "avg_weekly_demand",
        "days_of_stock",
        "unit_cost",
        "unit_price",
        "abc_class",
        "sku_count",
        "total_stock_qty",
        "below_safety_count",
        "total_ordered_qty",
    ]
    enriched_rows = []
    for row in base_entities_df.to_dict(orient="records"):
        eid = row["entity_id"]
        etype = row["entity_type"]
        record = {
            "entity_id": eid,
            "entity_name": row.get("entity_name") or eid,
            "entity_type": etype,
            "location": row.get("location") or "",
            "data_source": "cleaned_production_dataset",
        }

        if etype == "supplier" and eid in suppliers_idx.index:
            s = suppliers_idx.loc[eid]
            record["entity_name"] = str(s["supplier_name"])
            record["location"] = str(s["country"])
            record["category"] = str(s["category"])
            record["risk_tier"] = str(s["risk_tier"])
            record["is_preferred"] = str(bool(s["is_preferred"])).lower()
            record["avg_lead_time_days"] = str(s["avg_lead_time_days"])
            record["lead_time_variability"] = str(s["lead_time_variability"])
            record["annual_spend"] = str(s["annual_spend"])
            record["on_time_delivery_rate"] = str(s["on_time_delivery_rate"])
            record["quality_acceptance_rate"] = str(s["quality_acceptance_rate"])

        elif etype == "sku" and eid in inventory_idx.index:
            inv = inventory_idx.loc[eid]
            record["entity_name"] = str(inv["sku_name"])
            record["location"] = ""  # SKUs are non-spatial entities
            record["category"] = str(inv["category"])
            record["primary_supplier_id"] = str(inv["primary_supplier_id"])
            record["home_warehouse"] = str(inv["warehouse"])
            record["current_stock_qty"] = str(inv["current_stock_qty"])
            record["safety_stock_qty"] = str(inv["safety_stock_qty"])
            record["reorder_point_qty"] = str(inv["reorder_point_qty"])
            record["max_stock_qty"] = str(inv["max_stock_qty"])
            record["avg_weekly_demand"] = str(inv["avg_weekly_demand"])
            record["days_of_stock"] = str(inv["days_of_stock"])
            record["unit_cost"] = str(inv["unit_cost"])
            record["unit_price"] = str(inv["unit_price"])
            record["abc_class"] = str(inv["abc_class"])

        elif etype == "warehouse":
            record["entity_name"] = eid
            record["location"] = eid
            wh_stat = inv_wh_stats.get(eid, {})
            record["sku_count"] = str(wh_stat.get("sku_count", 0))
            record["total_stock_qty"] = str(wh_stat.get("total_stock_qty", 0))
            record["below_safety_count"] = str(wh_stat.get("below_safety_count", 0))
            record["total_ordered_qty"] = str(int(po_wh_vol.get(eid, 0)))

        enriched_rows.append(record)

    cols = REQUIRED_ENTITY_COLUMNS + ["data_source"] + extra_cols
    df = pd.DataFrame(enriched_rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].fillna("")


def build_enriched_tables_from_cleaned(
    tables: CleanedTables,
    backup_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    include_inventory_primary_links: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """Convert CleanedTables into validated (entities_df, relationships_df, report).

    Uses ``graph_builder.purchase_orders_to_tables`` for canonical dependency-share
    computation, then enriches entity metadata from ``suppliers.csv`` and
    ``inventory.csv``.
    """
    report = ValidationReport()
    po_str = tables.purchase_orders.astype(str)
    base_entities_df, relationships_df, report = purchase_orders_to_tables(
        po_str,
        report=report,
        columns=CLEANED_PO_COLUMN_MAP,
        backup_pairs=backup_pairs,
    )

    if include_inventory_primary_links:
        existing_pairs = set(
            zip(relationships_df["source_id"], relationships_df["target_id"])
        )
        extra_rels = []
        for row in tables.inventory.itertuples(index=False):
            pair = (str(row.primary_supplier_id), str(row.sku_id))
            if pair not in existing_pairs:
                extra_rels.append(
                    {
                        "source_id": pair[0],
                        "target_id": pair[1],
                        "relationship_type": "supplies",
                        "dependency_score": 0.0,
                        "material": pair[1],
                        "is_backup": "true",
                        "data_source": "inventory_primary_supplier",
                    }
                )
        if extra_rels:
            relationships_df = pd.concat(
                [relationships_df, pd.DataFrame(extra_rels)], ignore_index=True
            )

    entities_df = _enrich_entities(base_entities_df, tables)
    relationships_df["data_source"] = "cleaned_production_dataset"
    return entities_df, relationships_df, report


def adapt_disruption_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    """Adapt ``disruption_events_template.csv`` rows to ``graph_builder`` schema.

    Maps:
        event_id -> disruption_id
        affected_entity_id -> entity_id
        event_type -> disruption_type
    """
    df = events_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "event_id": "disruption_id",
        "affected_entity_id": "entity_id",
        "event_type": "disruption_type",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in ("location", "start_date", "duration_days"):
        if col not in df.columns:
            df[col] = ""
    keep = REQUIRED_DISRUPTION_COLUMNS + ["location", "start_date", "duration_days"]
    missing = [c for c in REQUIRED_DISRUPTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Disruption events missing required column(s): {', '.join(missing)}")
    return df[keep]


def load_full_supply_chain(
    data_dir: Optional[str | Path] = None,
    disruptions_csv: Optional[str | Path] = None,
    backup_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    strict: bool = True,
) -> SupplyChainData:
    """Build the full validated NetworkX DiGraph from the cleaned production CSVs."""
    tables = load_cleaned_tables(data_dir)
    entities_df, relationships_df, report = build_enriched_tables_from_cleaned(
        tables, backup_pairs=backup_pairs
    )

    disruptions_df = None
    if disruptions_csv is not None:
        raw_dis = pd.read_csv(disruptions_csv, dtype=str, keep_default_na=False)
        if not raw_dis.empty:
            if "event_id" in raw_dis.columns or "affected_entity_id" in raw_dis.columns:
                disruptions_df = adapt_disruption_events_df(raw_dis)
            else:
                disruptions_df = load_csv(
                    disruptions_csv, REQUIRED_DISRUPTION_COLUMNS, "disruptions"
                )

    return build_supply_chain(
        entities_df,
        relationships_df,
        disruptions_df=disruptions_df,
        strict=strict,
        report=report,
    )


def compute_executive_kpis(tables: CleanedTables, graph: nx.DiGraph) -> Dict:
    """Compute high-level supply chain, inventory, and bottleneck KPIs."""
    po = tables.purchase_orders
    ship = tables.shipments
    inv = tables.inventory
    sup = tables.suppliers

    below_safety = inv[inv["current_stock_qty"] < inv["safety_stock_qty"]]
    below_reorder = inv[inv["current_stock_qty"] <= inv["reorder_point_qty"]]
    low_dos = inv[inv["days_of_stock"] < 14.0]

    # Top vulnerable suppliers by downstream reach * historical/profile risk
    po_delay_by_sup = (
        po.assign(is_delayed=(~po["is_on_time"]).astype(int))
        .groupby("supplier_id")
        .agg(
            po_count=("po_id", "count"),
            delay_rate=("is_delayed", "mean"),
            avg_days_late=("days_late", "mean"),
            total_spend=("total_cost", "sum"),
        )
    )
    sup_merged = sup.set_index("supplier_id").join(po_delay_by_sup, how="left").fillna(0)
    bottlenecks = []
    for sid, row in sup_merged.iterrows():
        if sid not in graph:
            continue
        skus_served = graph.out_degree(sid)
        downstream = len(nx.descendants(graph, sid))
        vulnerability = round(
            float(row["delay_rate"]) * (1.0 + downstream / max(1, graph.number_of_nodes())) * 100,
            2,
        )
        bottlenecks.append(
            {
                "supplier_id": sid,
                "supplier_name": row["supplier_name"],
                "country": row["country"],
                "risk_tier": row["risk_tier"],
                "skus_supplied": skus_served,
                "downstream_nodes": downstream,
                "po_count": int(row["po_count"]),
                "observed_delay_rate": round(float(row["delay_rate"]), 4),
                "avg_days_late": round(float(row["avg_days_late"]), 2),
                "total_spend": round(float(row["total_spend"]), 2),
                "vulnerability_index": vulnerability,
            }
        )
    bottlenecks.sort(key=lambda x: (-x["vulnerability_index"], -x["downstream_nodes"]))

    return {
        "total_suppliers": int(sup["supplier_id"].nunique()),
        "total_skus": int(inv["sku_id"].nunique()),
        "total_warehouses": int(po["warehouse"].nunique()),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "total_pos": int(len(po)),
        "pending_pos": int((po["po_status"] == "pending").sum()),
        "late_pos": int((~po["is_on_time"]).sum()),
        "po_on_time_rate": round(float(po["is_on_time"].mean()), 4),
        "po_avg_days_late": round(float(po["days_late"].mean()), 2),
        "total_po_spend": round(float(po["total_cost"].sum()), 2),
        "total_shipments": int(len(ship)),
        "late_shipments": int((~ship["is_on_time"]).sum()),
        "shipment_on_time_rate": round(float(ship["is_on_time"].mean()), 4),
        "damaged_shipments": int(ship["damage_reported"].sum()),
        "below_safety_stock_skus": int(len(below_safety)),
        "below_reorder_point_skus": int(len(below_reorder)),
        "low_days_of_stock_skus": int(len(low_dos)),
        "avg_days_of_stock": round(float(inv["days_of_stock"].mean()), 1),
        "top_bottlenecks": bottlenecks[:15],
    }


if __name__ == "__main__":
    tables = load_cleaned_tables()
    sc = load_full_supply_chain(strict=True)
    kpis = compute_executive_kpis(tables, sc.graph)
    print(f"Loaded full supply chain graph: {sc.graph.number_of_nodes()} nodes, {sc.graph.number_of_edges()} edges")
    print(f"Validation valid={sc.report.is_valid}, issues={len(sc.report.issues)}")
    print("KPIs:", {k: v for k, v in kpis.items() if k != "top_bottlenecks"})
