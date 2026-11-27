"""
graph_builder.py
================
Loads the three CSV tables (entities, relationships, disruptions), validates
them, and builds a NetworkX directed graph.

Main entry points
-----------------
load_supply_chain(entities_csv, relationships_csv, disruptions_csv=None, strict=True)
build_supply_chain(entities_df, relationships_df, disruptions_df=None, strict=True)
graph_to_dict(graph)        -> plain dict/JSON view of the graph (for dashboards)

Purchase-order graph (Supplier -> SKU -> Warehouse)
load_supply_chain_from_purchase_orders(po_csv, disruptions_csv=None, columns=None, backup_pairs=None)
build_supply_chain_from_purchase_orders(po_df, ...)
    Relationships are derived only from the purchase-order rows (see the section below).

strict=True  : any data error raises GraphValidationError (safest default).
strict=False : rows with errors are dropped and listed in the validation report.

Run `python graph_builder.py --help` for a small command-line demo.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

from graph_utils import (
    ENTITY_TYPES,
    RELATIONSHIP_RULES,
    GraphValidationError,
    ValidationReport,
    describe_allowed_pairs,
    get_entity_ids_by_type,
    save_json,
)

# Columns that MUST exist in each CSV. Other columns are optional extras.
REQUIRED_ENTITY_COLUMNS = ["entity_id", "entity_name", "entity_type", "location"]
REQUIRED_RELATIONSHIP_COLUMNS = ["source_id", "target_id", "relationship_type", "dependency_score"]
REQUIRED_DISRUPTION_COLUMNS = ["disruption_id", "entity_id", "disruption_type", "severity"]
OPTIONAL_DISRUPTION_COLUMNS = ["location", "start_date", "duration_days"]


@dataclass
class SupplyChainData:
    """Everything produced by the loader."""
    graph: nx.DiGraph             # the supply-chain network
    disruptions: List[Dict]       # validated disruption records
    report: ValidationReport      # errors and warnings found in the data


# ---------------------------------------------------------------------------
# Small cleaning helpers
# ---------------------------------------------------------------------------

def _text(value) -> Optional[str]:
    """Return a stripped string, or None if the value is empty/missing."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned and cleaned.lower() != "nan" else None


def _to_float(value) -> Optional[float]:
    """Convert to a finite float, or None if that is not possible."""
    text = _text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _to_bool(value) -> Optional[bool]:
    """Parse true/false style values. Empty -> False. Unknown text -> None."""
    text = _text(value)
    if text is None:
        return False
    if text.lower() in ("true", "yes", "y", "1"):
        return True
    if text.lower() in ("false", "no", "n", "0"):
        return False
    return None


def _prepare_frame(df, required, table) -> pd.DataFrame:
    """Copy the DataFrame, normalise column names and check required columns."""
    if not isinstance(df, pd.DataFrame):
        raise GraphValidationError(f"{table}: expected a pandas DataFrame.")
    df = df.copy().reset_index(drop=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise GraphValidationError(f"{table}: missing required column(s): {', '.join(missing)}")
    return df


def load_csv(path, required, table) -> pd.DataFrame:
    """Read a CSV file (all values as text) and check its columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{table} file not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise GraphValidationError(f"{table}: the file {path} is empty.")
    return _prepare_frame(df, required, table)


# ---------------------------------------------------------------------------
# Validation of each table. Each returns a list of CLEAN records (dicts).
# "row" numbers in messages are CSV line numbers (header = line 1).
# ---------------------------------------------------------------------------

def validate_entities(df: pd.DataFrame, report: ValidationReport) -> List[Dict]:
    records, seen = [], {}
    extra_columns = [c for c in df.columns if c not in REQUIRED_ENTITY_COLUMNS]

    for idx, row in df.iterrows():
        line = idx + 2
        entity_id = _text(row["entity_id"])
        if entity_id is None:
            report.add_error("entities", line, "entity_id", "entity_id is missing.")
            continue
        if entity_id in seen:
            report.add_error("entities", line, "entity_id",
                             f"duplicate entity_id '{entity_id}' (first used on row {seen[entity_id]}).")
            continue

        entity_type = (_text(row["entity_type"]) or "").lower()
        if entity_type not in ENTITY_TYPES:
            report.add_error("entities", line, "entity_type",
                             f"'{row['entity_type']}' is not one of {list(ENTITY_TYPES)}.")
            continue

        name, location = _text(row["entity_name"]), _text(row["location"])
        if name is None:
            report.add_warning("entities", line, "entity_name", f"{entity_id} has no name.")
        if location is None and entity_type != "sku":   # a SKU is not a place
            report.add_warning("entities", line, "location", f"{entity_id} has no location.")

        record = {"entity_id": entity_id, "entity_name": name,
                  "entity_type": entity_type, "location": location}
        for col in extra_columns:            # keep any additional attributes
            value = _text(row[col])
            if value is not None:
                record[col] = value

        seen[entity_id] = line
        records.append(record)
    return records


def validate_relationships(df: pd.DataFrame, entities_by_id: Dict[str, Dict],
                           report: ValidationReport) -> List[Dict]:
    records, seen_pairs = [], {}
    has_backup_column = "is_backup" in df.columns
    has_material_column = "material" in df.columns

    for idx, row in df.iterrows():
        line = idx + 2
        errors_before = len(report.errors)

        source, target = _text(row["source_id"]), _text(row["target_id"])
        if source is None:
            report.add_error("relationships", line, "source_id", "source_id is missing.")
        elif source not in entities_by_id:
            report.add_error("relationships", line, "source_id", f"unknown entity '{source}'.")
        if target is None:
            report.add_error("relationships", line, "target_id", "target_id is missing.")
        elif target not in entities_by_id:
            report.add_error("relationships", line, "target_id", f"unknown entity '{target}'.")
        if source is not None and source == target:
            report.add_error("relationships", line, "target_id", "an entity cannot depend on itself.")

        rel_type = (_text(row["relationship_type"]) or "").lower()
        if rel_type not in RELATIONSHIP_RULES:
            report.add_error("relationships", line, "relationship_type",
                             f"'{row['relationship_type']}' is not one of {list(RELATIONSHIP_RULES)}.")
        elif source in entities_by_id and target in entities_by_id:
            actual = (entities_by_id[source]["entity_type"], entities_by_id[target]["entity_type"])
            if actual not in RELATIONSHIP_RULES[rel_type]:
                report.add_error("relationships", line, "relationship_type",
                                 f"'{rel_type}' must go {describe_allowed_pairs(rel_type)}, "
                                 f"but {source} is a {actual[0]} and {target} is a {actual[1]}.")

        dependency = _to_float(row["dependency_score"])
        if dependency is None:
            report.add_error("relationships", line, "dependency_score",
                             f"'{row['dependency_score']}' is not a number.")
        elif not 0.0 <= dependency <= 1.0:
            report.add_error("relationships", line, "dependency_score",
                             f"{dependency} is outside the allowed range 0 to 1.")

        is_backup = _to_bool(row["is_backup"]) if has_backup_column else False
        if is_backup is None:
            report.add_error("relationships", line, "is_backup",
                             f"'{row['is_backup']}' is not true/false.")

        if source and target and (source, target) in seen_pairs:
            report.add_error("relationships", line, "target_id",
                             f"duplicate relationship {source} -> {target} (first on row "
                             f"{seen_pairs[(source, target)]}). Combine materials into one row.")

        if len(report.errors) > errors_before:
            continue  # row has problems -> skip it

        material = _text(row["material"]) if has_material_column else None
        if (material is None and rel_type == "supplies"
                and entities_by_id[target]["entity_type"] == "sku"):
            # A SKU is its own "material": every supplier of the same SKU provides the same
            # thing, so a backup supplier of a SKU matches its normal suppliers by default.
            material = target
        seen_pairs[(source, target)] = line
        records.append({"source_id": source, "target_id": target, "relationship_type": rel_type,
                        "dependency_score": dependency, "material": material,
                        "is_backup": bool(is_backup), "_row": line})

    _check_relationship_consistency(records, report)
    return records


def _check_relationship_consistency(records: List[Dict], report: ValidationReport) -> None:
    """Whole-table checks: share totals, backup suppliers, and cycles."""
    # Total dependency on normal (non-backup) sources should not exceed 1.
    totals: Dict[str, float] = {}
    for r in records:
        if not r["is_backup"]:
            totals[r["target_id"]] = totals.get(r["target_id"], 0.0) + r["dependency_score"]
    for target, total in sorted(totals.items()):
        if total > 1.0 + 1e-9:
            report.add_warning("relationships", None, "dependency_score",
                               f"dependency scores into '{target}' add up to {total:.2f} (> 1). "
                               f"Impact on this entity is capped at 1.")

    # Backup suppliers only help if their material matches a normal supplier's material.
    primary_materials = {(r["target_id"], (r["material"] or "").lower())
                         for r in records if not r["is_backup"] and r["material"]}
    for r in records:
        if not r["is_backup"]:
            continue
        key = (r["target_id"], (r["material"] or "").lower())
        if not r["material"]:
            report.add_warning("relationships", r["_row"], "material",
                               "backup supplier has no material, so it cannot reduce any impact.")
        elif key not in primary_materials:
            report.add_warning("relationships", r["_row"], "material",
                               f"no normal supplier of '{r['material']}' to {r['target_id']}; "
                               f"this backup will have no effect.")

    # A supply chain should flow one way. A cycle would make propagation ambiguous.
    flow = nx.DiGraph([(r["source_id"], r["target_id"]) for r in records if not r["is_backup"]])
    if not nx.is_directed_acyclic_graph(flow):
        cycle = [edge[0] for edge in nx.find_cycle(flow)]
        message = "the relationships contain a cycle: " + " -> ".join(cycle + [cycle[0]])
        report.add_error("relationships", None, None, message)
        raise GraphValidationError(message, report)  # cannot be fixed by dropping one row


def validate_disruptions(df: pd.DataFrame, entities_by_id: Dict[str, Dict],
                         report: ValidationReport) -> List[Dict]:
    for col in OPTIONAL_DISRUPTION_COLUMNS:
        if col not in df.columns:
            report.add_warning("disruptions", None, col, f"column '{col}' is missing; values will be empty.")
            df[col] = ""

    records, seen = [], {}
    for idx, row in df.iterrows():
        line = idx + 2
        errors_before = len(report.errors)

        disruption_id = _text(row["disruption_id"])
        if disruption_id is None:
            report.add_error("disruptions", line, "disruption_id", "disruption_id is missing.")
        elif disruption_id in seen:
            report.add_error("disruptions", line, "disruption_id",
                             f"duplicate disruption_id '{disruption_id}' (first on row {seen[disruption_id]}).")

        entity_id = _text(row["entity_id"])
        if entity_id is None:
            report.add_error("disruptions", line, "entity_id", "entity_id is missing.")
        elif entity_id not in entities_by_id:
            report.add_error("disruptions", line, "entity_id", f"unknown entity '{entity_id}'.")

        disruption_type = _text(row["disruption_type"])
        if disruption_type is None:
            report.add_error("disruptions", line, "disruption_type", "disruption_type is missing.")

        severity = _to_float(row["severity"])
        if severity is None:
            report.add_error("disruptions", line, "severity", f"'{row['severity']}' is not a number.")
        elif not 0.0 <= severity <= 1.0:
            report.add_error("disruptions", line, "severity", f"{severity} is outside the range 0 to 1.")

        start_date = None
        raw_date = _text(row["start_date"])
        if raw_date is None:
            report.add_warning("disruptions", line, "start_date", "start_date is empty.")
        else:
            parsed = pd.to_datetime(raw_date, errors="coerce")
            if pd.isna(parsed):
                report.add_error("disruptions", line, "start_date", f"'{raw_date}' is not a valid date.")
            else:
                start_date = parsed.date().isoformat()

        duration = None
        raw_duration = _text(row["duration_days"])
        if raw_duration is not None:
            duration = _to_float(raw_duration)
            if duration is None or duration < 0:
                report.add_error("disruptions", line, "duration_days",
                                 f"'{raw_duration}' is not a non-negative number.")
            elif duration == int(duration):
                duration = int(duration)

        location = _text(row["location"])
        if location is None:
            report.add_warning("disruptions", line, "location", "location is empty.")

        if len(report.errors) > errors_before:
            continue

        seen[disruption_id] = line
        records.append({"disruption_id": disruption_id, "entity_id": entity_id,
                        "disruption_type": disruption_type, "severity": severity,
                        "location": location, "start_date": start_date,
                        "duration_days": duration})
    return records


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(entity_records: List[Dict], relationship_records: List[Dict]) -> nx.DiGraph:
    """Create the DiGraph from CLEAN records (use build_supply_chain for raw data).

    Edge direction: source -> target means "target depends on source".
    """
    graph = nx.DiGraph()
    for record in entity_records:
        attributes = {k: v for k, v in record.items() if k != "entity_id"}
        graph.add_node(record["entity_id"], **attributes)
    for r in relationship_records:
        graph.add_edge(r["source_id"], r["target_id"],
                       relationship_type=r["relationship_type"],
                       dependency_score=r["dependency_score"],
                       material=r["material"],
                       is_backup=r["is_backup"])
    return graph


def build_supply_chain(entities_df, relationships_df, disruptions_df=None, strict=True,
                       report: Optional[ValidationReport] = None) -> SupplyChainData:
    """Validate DataFrames and build the graph.

    strict=True  -> raise GraphValidationError if any data error exists.
    strict=False -> drop bad rows, keep going, and list the problems in the report.
    report       -> optional existing report to add to (used by the purchase-order loader).
    """
    if report is None:
        report = ValidationReport()

    entities_df = _prepare_frame(entities_df, REQUIRED_ENTITY_COLUMNS, "entities")
    relationships_df = _prepare_frame(relationships_df, REQUIRED_RELATIONSHIP_COLUMNS, "relationships")

    entity_records = validate_entities(entities_df, report)
    if not entity_records:
        raise GraphValidationError("No valid entities found.\n" + report.summary(), report)
    entities_by_id = {e["entity_id"]: e for e in entity_records}

    relationship_records = validate_relationships(relationships_df, entities_by_id, report)
    if not relationship_records:
        report.add_warning("relationships", None, None, "no valid relationships; the graph has no edges.")

    disruption_records = []
    if disruptions_df is not None:
        disruptions_df = _prepare_frame(disruptions_df, REQUIRED_DISRUPTION_COLUMNS, "disruptions")
        disruption_records = validate_disruptions(disruptions_df, entities_by_id, report)

    if strict and not report.is_valid:
        raise GraphValidationError(report.summary(), report)

    # Remove the internal helper field before building the graph.
    graph = build_graph(entity_records, [{k: v for k, v in r.items() if k != "_row"}
                                         for r in relationship_records])
    return SupplyChainData(graph=graph, disruptions=disruption_records, report=report)


def load_supply_chain(entities_csv, relationships_csv, disruptions_csv=None, strict=True) -> SupplyChainData:
    """Build the graph straight from CSV files (the usual way to use this module)."""
    entities_df = load_csv(entities_csv, REQUIRED_ENTITY_COLUMNS, "entities")
    relationships_df = load_csv(relationships_csv, REQUIRED_RELATIONSHIP_COLUMNS, "relationships")
    disruptions_df = None
    if disruptions_csv is not None:
        disruptions_df = load_csv(disruptions_csv, REQUIRED_DISRUPTION_COLUMNS, "disruptions")
    return build_supply_chain(entities_df, relationships_df, disruptions_df, strict=strict)


# ---------------------------------------------------------------------------
# Supplier -> SKU -> Warehouse graph built from purchase-order records
# ---------------------------------------------------------------------------
# The relationships come ONLY from what the purchase orders say. IDs are used exactly
# as they appear in the data; nothing is mapped, renamed or matched to other ID schemes,
# and no supplier-to-factory link is invented.
#
#   supplier --supplies--> sku       dependency_score = the supplier's share of the SKU's
#                                    normal (non-backup) ordered volume
#   sku --ships_to--> warehouse      dependency_score = the SKU's share of the warehouse's
#                                    received volume
#
# "Volume" is the quantity column; without one, each PO line counts as 1.
# The shares into one SKU (normal suppliers) and into one warehouse each add up to 1.

PO_TABLE = "purchase_orders"

# logical field -> default column name in the purchase-order table.
# Pass columns={"supplier_id": "vendor_code", ...} to match your dataset's real headers.
DEFAULT_PO_COLUMNS = {
    "supplier_id": "supplier_id",              # required
    "sku_id": "sku_id",                        # required
    "warehouse_id": "warehouse_id",            # required
    "quantity": "quantity",                    # optional (default: 1 per PO line)
    "is_backup": "is_backup",                  # optional: true on backup-supplier orders
    "supplier_name": "supplier_name",          # optional
    "supplier_location": "supplier_location",  # optional
    "sku_name": "sku_name",                    # optional
    "warehouse_name": "warehouse_name",        # optional
    "warehouse_location": "warehouse_location",  # optional
}
REQUIRED_PO_FIELDS = ("supplier_id", "sku_id", "warehouse_id")


def _resolve_po_columns(df: pd.DataFrame, columns: Optional[Dict[str, str]]) -> Dict[str, Optional[str]]:
    """Map each logical field to a column present in df (or None if optional and absent)."""
    names = dict(DEFAULT_PO_COLUMNS)
    for field, column in (columns or {}).items():
        if field not in DEFAULT_PO_COLUMNS:
            raise GraphValidationError(f"{PO_TABLE}: unknown column mapping '{field}'. "
                                       f"Allowed: {', '.join(DEFAULT_PO_COLUMNS)}.")
        names[field] = str(column)
    names = {field: column.strip().lower() for field, column in names.items()}
    missing = [f"{f} (column '{names[f]}')" for f in REQUIRED_PO_FIELDS if names[f] not in df.columns]
    if missing:
        raise GraphValidationError(f"{PO_TABLE}: missing required column(s): {', '.join(missing)}. "
                                   f"Use the `columns` argument to map your column names.")
    return {field: (column if column in df.columns else None) for field, column in names.items()}


def purchase_orders_to_tables(po_df: pd.DataFrame, report: Optional[ValidationReport] = None,
                              columns: Optional[Dict[str, str]] = None,
                              backup_pairs: Optional[Iterable[Tuple[str, str]]] = None):
    """Turn purchase-order rows into (entities_df, relationships_df, report).

    The two DataFrames use the normal entities / relationships layout, so they go
    through the same validation as hand-written CSVs (see build_supply_chain).

    backup_pairs: optional (supplier_id, sku_id) pairs known to be BACKUP suppliers.
    A backup can also be marked with a true/false `is_backup` column on the PO rows.
    Purchase orders alone do not say who is a backup, so nobody is one by default.
    A backup's dependency_score is its ordered volume relative to the SKU's normal
    volume (capped at 1): the share of the normal supply it has shown it can replace.
    """
    report = report if report is not None else ValidationReport()
    po_df = _prepare_frame(po_df, [], PO_TABLE)
    col = _resolve_po_columns(po_df, columns)
    backup_set = {(str(s).strip(), str(k).strip()) for s, k in (backup_pairs or [])}

    if col["quantity"] is None:
        report.add_warning(PO_TABLE, None, "quantity",
                           "no quantity column; every PO line counts as 1 unit when computing shares.")

    # --- read rows -------------------------------------------------------------------
    rows = []                                   # (line, supplier, sku, warehouse, volume, backup)
    attrs: Dict[str, Dict[str, Dict]] = {"supplier": {}, "sku": {}, "warehouse": {}}
    attr_columns = {"supplier": ("supplier_name", "supplier_location"), "sku": ("sku_name", None),
                    "warehouse": ("warehouse_name", "warehouse_location")}
    id_fields = {"supplier": "supplier_id", "sku": "sku_id", "warehouse": "warehouse_id"}

    for idx, row in po_df.iterrows():
        line = idx + 2
        errors_before = len(report.errors)
        ids = {}
        for kind, field in id_fields.items():
            ids[kind] = _text(row[col[field]])
            if ids[kind] is None:
                report.add_error(PO_TABLE, line, col[field], f"{field} is missing.")

        volume = 1.0
        if col["quantity"] is not None:
            volume = _to_float(row[col["quantity"]])
            if volume is None or volume <= 0:
                report.add_error(PO_TABLE, line, col["quantity"],
                                 f"'{row[col['quantity']]}' is not a positive number.")

        is_backup = _to_bool(row[col["is_backup"]]) if col["is_backup"] else False
        if is_backup is None:
            report.add_error(PO_TABLE, line, col["is_backup"], f"'{row[col['is_backup']]}' is not true/false.")
        if len(report.errors) > errors_before:
            continue                            # row has problems -> skip it

        if (ids["supplier"], ids["sku"]) in backup_set:
            is_backup = True
        rows.append((line, ids["supplier"], ids["sku"], ids["warehouse"], volume, bool(is_backup)))

        for kind, (name_field, loc_field) in attr_columns.items():
            record = attrs[kind].setdefault(ids[kind], {"entity_name": None, "location": None})
            for key, field in (("entity_name", name_field), ("location", loc_field)):
                value = _text(row[col[field]]) if field and col[field] else None
                if value is None:
                    continue
                if record[key] is None:
                    record[key] = value
                elif record[key] != value:
                    report.add_warning(PO_TABLE, line, col[field],
                                       f"{ids[kind]} has more than one {key} ('{record[key]}' and "
                                       f"'{value}'); using the first.")

    # --- volumes ---------------------------------------------------------------------
    pair_rows: Dict[Tuple[str, str], List[Tuple[float, bool]]] = {}
    sku_wh_volume: Dict[Tuple[str, str], float] = {}
    warehouse_total: Dict[str, float] = {}
    for _, supplier, sku, warehouse, volume, backup in rows:
        pair_rows.setdefault((supplier, sku), []).append((volume, backup))
        sku_wh_volume[(sku, warehouse)] = sku_wh_volume.get((sku, warehouse), 0.0) + volume
        warehouse_total[warehouse] = warehouse_total.get(warehouse, 0.0) + volume

    normal_volume: Dict[Tuple[str, str], float] = {}   # (supplier, sku) -> volume as NORMAL supplier
    backup_volume: Dict[Tuple[str, str], float] = {}
    for pair, items in sorted(pair_rows.items()):
        normal = sum(v for v, b in items if not b)
        backup = sum(v for v, b in items if b)
        if normal > 0:
            normal_volume[pair] = normal
            if backup > 0:
                report.add_warning(PO_TABLE, None, None,
                                   f"{pair[0]} has both normal and backup orders for {pair[1]}; "
                                   f"treated as a normal supplier of it.")
        else:
            backup_volume[pair] = backup

    sku_normal_total: Dict[str, float] = {}
    for (supplier, sku), volume in normal_volume.items():
        sku_normal_total[sku] = sku_normal_total.get(sku, 0.0) + volume
    for (supplier, sku), volume in sorted(backup_volume.items()):
        if sku not in sku_normal_total:        # a backup of nothing -> treat as the normal source
            report.add_warning(PO_TABLE, None, None,
                               f"{supplier} is marked as a backup for {sku}, but {sku} has no normal "
                               f"supplier; treated as a normal supplier.")
            normal_volume[(supplier, sku)] = volume
            sku_normal_total[sku] = sku_normal_total.get(sku, 0.0) + volume
    backup_volume = {p: v for p, v in backup_volume.items() if p not in normal_volume}

    # --- tables ----------------------------------------------------------------------
    entity_rows = []
    for kind in ("supplier", "sku", "warehouse"):
        for entity_id in sorted(attrs[kind]):
            record = attrs[kind][entity_id]
            entity_rows.append({"entity_id": entity_id, "entity_name": record["entity_name"] or entity_id,
                                "entity_type": kind, "location": record["location"] or "",
                                "data_source": PO_TABLE})

    rel_rows = []
    for (supplier, sku), volume in sorted(normal_volume.items()):
        rel_rows.append({"source_id": supplier, "target_id": sku, "relationship_type": "supplies",
                         "dependency_score": volume / sku_normal_total[sku], "material": sku,
                         "is_backup": "false", "data_source": PO_TABLE})
    for (supplier, sku), volume in sorted(backup_volume.items()):
        rel_rows.append({"source_id": supplier, "target_id": sku, "relationship_type": "supplies",
                         "dependency_score": min(1.0, volume / sku_normal_total[sku]), "material": sku,
                         "is_backup": "true", "data_source": PO_TABLE})
    for (sku, warehouse), volume in sorted(sku_wh_volume.items()):
        rel_rows.append({"source_id": sku, "target_id": warehouse, "relationship_type": "ships_to",
                         "dependency_score": volume / warehouse_total[warehouse], "material": sku,
                         "is_backup": "false", "data_source": PO_TABLE})

    entities_df = pd.DataFrame(entity_rows, columns=REQUIRED_ENTITY_COLUMNS + ["data_source"])
    relationships_df = pd.DataFrame(rel_rows, columns=REQUIRED_RELATIONSHIP_COLUMNS +
                                    ["material", "is_backup", "data_source"])
    return entities_df, relationships_df, report


def build_supply_chain_from_purchase_orders(po_df, disruptions_df=None, columns=None,
                                            backup_pairs=None, strict=True) -> SupplyChainData:
    """Purchase-order DataFrame -> validated Supplier -> SKU -> Warehouse graph.

    Same strict / lenient behaviour as build_supply_chain; problems found in the PO rows
    are reported under the table name 'purchase_orders'.
    """
    report = ValidationReport()
    entities_df, relationships_df, report = purchase_orders_to_tables(po_df, report, columns, backup_pairs)
    return build_supply_chain(entities_df, relationships_df, disruptions_df, strict=strict, report=report)


def load_supply_chain_from_purchase_orders(purchase_orders_csv, disruptions_csv=None, columns=None,
                                           backup_pairs=None, strict=True) -> SupplyChainData:
    """Build the Supplier -> SKU -> Warehouse graph straight from a purchase-order CSV."""
    path = Path(purchase_orders_csv)
    if not path.is_file():
        raise FileNotFoundError(f"{PO_TABLE} file not found: {path}")
    try:
        po_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise GraphValidationError(f"{PO_TABLE}: the file {path} is empty.")
    disruptions_df = None
    if disruptions_csv is not None:
        disruptions_df = load_csv(disruptions_csv, REQUIRED_DISRUPTION_COLUMNS, "disruptions")
    return build_supply_chain_from_purchase_orders(po_df, disruptions_df, columns, backup_pairs, strict)


# ---------------------------------------------------------------------------
# Export helpers (for the dashboard / other modules)
# ---------------------------------------------------------------------------

def graph_to_dict(graph: nx.DiGraph) -> Dict:
    """Plain, JSON-friendly description of the graph: entities + relationships."""
    entities = [{"entity_id": node, **data} for node, data in graph.nodes(data=True)]
    relationships = [{"source_id": u, "target_id": v, **data} for u, v, data in graph.edges(data=True)]
    return {"entities": entities, "relationships": relationships}


def graph_summary(graph: nx.DiGraph) -> Dict:
    """Quick counts, handy for logging or a dashboard header."""
    return {
        "total_entities": graph.number_of_nodes(),
        "suppliers": len(get_entity_ids_by_type(graph, "supplier")),
        "factories": len(get_entity_ids_by_type(graph, "factory")),
        "skus": len(get_entity_ids_by_type(graph, "sku")),
        "warehouses": len(get_entity_ids_by_type(graph, "warehouse")),
        "total_relationships": graph.number_of_edges(),
        "backup_relationships": sum(1 for _, _, d in graph.edges(data=True) if d.get("is_backup")),
    }


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build and validate the supply-chain graph.")
    parser.add_argument("--data-dir", default=str(here / "data"),
                        help="folder containing entities.csv, relationships.csv, disruptions.csv")
    parser.add_argument("--purchase-orders", help="build the Supplier -> SKU -> Warehouse graph from this "
                                                  "purchase-order CSV instead of the entity/relationship CSVs")
    parser.add_argument("--export", help="optional path to write the graph as JSON")
    parser.add_argument("--lenient", action="store_true", help="drop invalid rows instead of failing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    try:
        if args.purchase_orders:
            disruptions = data_dir / "disruptions.csv"
            data = load_supply_chain_from_purchase_orders(
                args.purchase_orders, disruptions if disruptions.is_file() else None,
                strict=not args.lenient)
        else:
            data = load_supply_chain(data_dir / "entities.csv", data_dir / "relationships.csv",
                                     data_dir / "disruptions.csv", strict=not args.lenient)
    except (FileNotFoundError, GraphValidationError) as error:
        raise SystemExit(f"Could not build the graph:\n{error}")

    print("Graph built successfully.")
    for key, value in graph_summary(data.graph).items():
        print(f"  {key}: {value}")
    print(f"  disruptions loaded: {len(data.disruptions)}")
    if data.report.issues:
        print(data.report.summary())
    if args.export:
        print(f"Graph JSON written to {save_json(graph_to_dict(data.graph), args.export)}")


if __name__ == "__main__":
    main()
