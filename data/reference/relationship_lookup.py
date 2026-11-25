"""Look up recorded supplier -> SKU -> warehouse relationships.

This uses only relationships present in the supplied CSVs. It does not infer
disruption events, factories, or causal propagation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_SCHEMA_VERSION = "1.0"


def retrieve_related_entities(
    supplier_id: str,
    *,
    sku_id: str | None = None,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Return recorded SKUs and warehouses associated with ``supplier_id``.

    Historical purchase orders establish supplier/SKU/warehouse links; current
    inventory establishes primary-supplier/SKU/warehouse links. Results are
    deduplicated and tagged with their evidence source.
    """
    root = Path(data_dir)
    po = pd.read_csv(root / "purchase_orders.csv", dtype=str)
    inv = pd.read_csv(root / "inventory.csv", dtype=str)
    suppliers = pd.read_csv(root / "suppliers.csv", dtype=str)

    if not (suppliers["supplier_id"] == str(supplier_id)).any():
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "query": {"entity_type": "supplier", "entity_id": str(supplier_id), "sku_id": sku_id},
            "status": "not_found",
            "coverage": _coverage(),
            "related_entities": [],
        }

    po_rows = po.loc[po.supplier_id == str(supplier_id)]
    inv_rows = inv.loc[inv.primary_supplier_id == str(supplier_id)]
    if sku_id is not None:
        po_rows = po_rows.loc[po_rows.sku_id == str(sku_id)]
        inv_rows = inv_rows.loc[inv_rows.sku_id == str(sku_id)]

    edges: dict[tuple[str, str], set[str]] = {}
    for row in po_rows.itertuples(index=False):
        edges.setdefault((row.sku_id, row.warehouse), set()).add("purchase_order")
    for row in inv_rows.itertuples(index=False):
        edges.setdefault((row.sku_id, row.warehouse), set()).add("inventory_primary_supplier")

    sku_names = inv.drop_duplicates("sku_id").set_index("sku_id")["sku_name"].to_dict()
    related = [
        {
            "entity_type": "warehouse",
            "entity_id": warehouse,
            "relationship": "supplier_supplies_sku_to_warehouse",
            "intermediate_entity": {"entity_type": "sku", "entity_id": sku, "name": sku_names.get(sku)},
            "evidence_sources": sorted(sources),
        }
        for (sku, warehouse), sources in sorted(edges.items())
    ]
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "query": {"entity_type": "supplier", "entity_id": str(supplier_id), "sku_id": sku_id},
        "status": "ok" if related else "no_recorded_relationships",
        "coverage": _coverage(),
        "related_entities": related,
    }


def _coverage() -> dict[str, Any]:
    return {
        "available_paths": ["supplier -> SKU -> warehouse"],
        "available_event_history": False,
        "factory_entities_available": False,
        "disruption_event_data_available": False,
        "limitations": [
            "Warehouse names are used as identifiers because stable warehouse IDs are absent.",
            "Relationships describe recorded orders and primary-supplier inventory, not disruption causality.",
            "No factories, sub-tier dependencies, or disruption propagation paths are present.",
        ],
    }
