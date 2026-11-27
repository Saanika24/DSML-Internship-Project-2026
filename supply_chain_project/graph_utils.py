"""
graph_utils.py
==============
Shared building blocks for the Graph Construction module.

Contents
--------
1. Constants (allowed entity types, relationship rules, risk thresholds)
2. Custom exceptions
3. ValidationReport  - collects data problems found while loading CSV files
4. Reusable retrieval functions - "who supplies whom?", "what is downstream?"
5. save_json helper

Other modules (graph_builder.py, impact_analysis.py) import from here.
Teammates can also import the retrieval functions directly.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

ENTITY_TYPES = ("supplier", "factory", "warehouse", "sku")

# relationship_type -> tuple of allowed (source entity type, target entity type) pairs.
# To support new relationship kinds later, add a row here.
#
# The purchase-order dataset has supplier, SKU and warehouse IDs but no verified
# supplier-to-factory mapping, so the real graph is  Supplier -> SKU -> Warehouse.
# The original Supplier -> Factory -> Warehouse pairs stay valid, so existing data
# and code keep working.
RELATIONSHIP_RULES = {
    "supplies": (("supplier", "sku"),          # supplier -> sku        (purchase-order graph)
                 ("supplier", "factory")),     # supplier -> factory    (original schema)
    "ships_to": (("sku", "warehouse"),         # sku      -> warehouse  (purchase-order graph)
                 ("factory", "warehouse")),    # factory  -> warehouse  (original schema)
}


def describe_allowed_pairs(relationship_type: str) -> str:
    """Readable list of the allowed pairs, e.g. 'supplier -> sku or supplier -> factory'."""
    return " or ".join(f"{s} -> {t}" for s, t in RELATIONSHIP_RULES[relationship_type])

# Risk level thresholds applied to an impact score between 0 and 1.
HIGH_RISK_THRESHOLD = 0.70    # score >= 0.70 -> "high"
MEDIUM_RISK_THRESHOLD = 0.40  # score >= 0.40 -> "medium"; above 0 -> "low"

SCORE_DECIMALS = 4            # impact scores are rounded to this many decimals
SCHEMA_VERSION = "1.1"        # version of the JSON output format (1.1: time windows)


# ---------------------------------------------------------------------------
# 2. Custom exceptions
# ---------------------------------------------------------------------------

class GraphValidationError(ValueError):
    """Raised when the input data is invalid (bad columns, bad values, ...)."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report  # optional ValidationReport with all details


class EntityNotFoundError(ValueError):
    """Raised when an entity_id is not present in the graph."""


# ---------------------------------------------------------------------------
# 3. Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    level: str            # "error" or "warning"
    table: str            # "entities", "relationships" or "disruptions"
    row: Optional[int]    # CSV line number (header is line 1), None if general
    field: Optional[str]  # column name, if relevant
    message: str


class ValidationReport:
    """Collects errors (must fix) and warnings (worth knowing) about the data."""

    def __init__(self):
        self.issues: List[ValidationIssue] = []

    def add_error(self, table, row, field, message):
        self.issues.append(ValidationIssue("error", table, row, field, message))

    def add_warning(self, table, row, field, message):
        self.issues.append(ValidationIssue("warning", table, row, field, message))

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [asdict(i) for i in self.issues],
        }

    def summary(self) -> str:
        lines = [f"Validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for issue in self.issues:
            where = issue.table
            if issue.row is not None:
                where += f" row {issue.row}"
            if issue.field:
                where += f", field '{issue.field}'"
            lines.append(f"  [{issue.level.upper()}] {where}: {issue.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Reusable retrieval functions
# ---------------------------------------------------------------------------
# The graph is a networkx.DiGraph:
#   node id   = entity_id, node attributes = entity_name, entity_type, location, ...
#   edge u->v = "v depends on u", edge attributes = relationship_type,
#               dependency_score, material, is_backup
# "Backup" edges describe alternative suppliers. They do NOT carry normal
# material flow, so most functions ignore them unless include_backup=True.

def check_entity_exists(graph: nx.DiGraph, entity_id: str) -> None:
    """Raise EntityNotFoundError if entity_id is not a node of the graph."""
    if entity_id not in graph:
        raise EntityNotFoundError(f"Entity '{entity_id}' was not found in the graph.")


def get_flow_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of the graph WITHOUT backup edges (the normal flow network)."""
    flow = nx.DiGraph()
    flow.add_nodes_from(graph.nodes(data=True))
    flow.add_edges_from(
        (u, v, d) for u, v, d in graph.edges(data=True) if not d.get("is_backup", False)
    )
    return flow


def get_entity_ids_by_type(graph: nx.DiGraph, entity_type: str) -> List[str]:
    """All entity ids of one type, e.g. every 'factory'."""
    return sorted(n for n, d in graph.nodes(data=True) if d.get("entity_type") == entity_type)


def get_downstream_entities(graph, entity_id, entity_type=None, include_backup=False) -> List[str]:
    """Every entity that can be reached from entity_id by following the flow.

    entity_type: optionally keep only 'supplier', 'factory', 'sku' or 'warehouse'.
    """
    check_entity_exists(graph, entity_id)
    source_graph = graph if include_backup else get_flow_graph(graph)
    found = nx.descendants(source_graph, entity_id)
    return _filter_and_sort(graph, found, entity_type)


def get_upstream_entities(graph, entity_id, entity_type=None, include_backup=False) -> List[str]:
    """Every entity that entity_id (directly or indirectly) depends on."""
    check_entity_exists(graph, entity_id)
    source_graph = graph if include_backup else get_flow_graph(graph)
    found = nx.ancestors(source_graph, entity_id)
    return _filter_and_sort(graph, found, entity_type)


def get_neighbors(graph, entity_id, direction="out", entity_type=None, include_backup=False) -> List[str]:
    """Directly connected entities.

    direction="out": entities that entity_id sends to (its customers).
    direction="in" : entities that send to entity_id (its sources).
    """
    check_entity_exists(graph, entity_id)
    if direction == "out":
        edges = graph.out_edges(entity_id, data=True)
    elif direction == "in":
        edges = graph.in_edges(entity_id, data=True)
    else:
        raise ValueError("direction must be 'in' or 'out'")

    result = []
    for u, v, data in edges:
        if data.get("is_backup", False) and not include_backup:
            continue
        other = v if direction == "out" else u
        if entity_type and graph.nodes[other].get("entity_type") != entity_type:
            continue
        result.append(other)
    return sorted(result)


# Friendly wrappers for the most common questions -----------------------------

def get_suppliers_of_factory(graph, factory_id, include_backup=False) -> List[str]:
    """Which suppliers supply materials to this factory?"""
    return get_neighbors(graph, factory_id, "in", "supplier", include_backup)


def get_factories_supplied_by(graph, supplier_id) -> List[str]:
    """Which factories does this supplier supply?"""
    return get_neighbors(graph, supplier_id, "out", "factory")


def get_warehouses_served_by(graph, factory_id) -> List[str]:
    """Which warehouses does this factory serve?"""
    return get_neighbors(graph, factory_id, "out", "warehouse")


def get_factories_serving(graph, warehouse_id) -> List[str]:
    """Which factories serve this warehouse?"""
    return get_neighbors(graph, warehouse_id, "in", "factory")


# Supplier -> SKU -> Warehouse helpers (purchase-order graph) ------------------

def get_suppliers_of_sku(graph, sku_id, include_backup=False) -> List[str]:
    """Which suppliers supply this SKU? (normal suppliers; backups only if include_backup)"""
    return get_neighbors(graph, sku_id, "in", "supplier", include_backup)


def get_backup_suppliers_of_sku(graph, sku_id) -> List[str]:
    """Suppliers listed as BACKUP (alternative) sources of this SKU."""
    check_entity_exists(graph, sku_id)
    return sorted(u for u, _, d in graph.in_edges(sku_id, data=True)
                  if d.get("is_backup", False) and graph.nodes[u].get("entity_type") == "supplier")


def get_skus_supplied_by(graph, supplier_id, include_backup=False) -> List[str]:
    """Which SKUs does this supplier supply? (include_backup adds SKUs it is only a backup for)"""
    return get_neighbors(graph, supplier_id, "out", "sku", include_backup)


def get_warehouses_of_sku(graph, sku_id) -> List[str]:
    """Which warehouses receive this SKU?"""
    return get_neighbors(graph, sku_id, "out", "warehouse")


def get_skus_at_warehouse(graph, warehouse_id) -> List[str]:
    """Which SKUs flow into this warehouse?"""
    return get_neighbors(graph, warehouse_id, "in", "sku")


def get_connected_entities(graph, entity_id) -> Dict[str, List[str]]:
    """Everything connected to entity_id: {'upstream': [...], 'downstream': [...]}."""
    return {
        "upstream": get_upstream_entities(graph, entity_id),
        "downstream": get_downstream_entities(graph, entity_id),
    }


def _filter_and_sort(graph, entity_ids, entity_type) -> List[str]:
    if entity_type:
        entity_ids = [e for e in entity_ids if graph.nodes[e].get("entity_type") == entity_type]
    return sorted(entity_ids)


# ---------------------------------------------------------------------------
# Risk level helper
# ---------------------------------------------------------------------------

def risk_level(score: float, high=HIGH_RISK_THRESHOLD, medium=MEDIUM_RISK_THRESHOLD) -> str:
    """Convert an impact score (0-1) to 'high', 'medium', 'low' or 'none'."""
    score = round(score, SCORE_DECIMALS)
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    if score > 0:
        return "low"
    return "none"


# ---------------------------------------------------------------------------
# 5. JSON helper
# ---------------------------------------------------------------------------

def save_json(data, path) -> Path:
    """Write data to a pretty-printed UTF-8 JSON file (folders are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
