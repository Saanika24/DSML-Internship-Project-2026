"""
impact_analysis.py
==================
Disruption propagation and dependency-based impact scoring on the supply-chain graph.

How the impact score works
--------------------------
Every edge u -> v has a dependency_score: the share (0-1) of v's operation that
depends on u. The graph can be Supplier -> Factory -> Warehouse, or (built from
purchase orders) Supplier -> SKU -> Warehouse; the maths only looks at edges, so it
is the same for both. A disruption of severity s on entity X gives X an impact of s.
For every entity v downstream of X (processed in dependency order):

    impact(v) = min( 1,  sum over normal sources u of
                         impact(u) * dependency_score(u, v) * (1 - mitigation(u, v)) )

* A supplier disruption does NOT automatically stop a factory or SKU: only the
  dependency share is lost, so impact(v) is usually below the severity. When several
  normal suppliers feed the same SKU, each one only carries its own share.
* mitigation(u, v) accounts for alternative suppliers. If the dataset lists a
  backup supplier (is_backup = true) of the SAME material to v, and that backup
  is not itself hit by this disruption, mitigation = the backup's
  dependency_score (the share of u's volume the backup can replace).
  Without such a backup, mitigation = 0. For a SKU, "same material" means the same
  SKU (edges into a SKU default their material to the SKU id).

Time windows and concurrent disruptions
---------------------------------------
Each disruption covers the days  start_date .. start_date + duration_days - 1
(inclusive). duration_days empty = open-ended (still running); start_date empty
= timing unknown, so the disruption is conservatively treated as overlapping
everything. A duration of 0 is treated as a single day.

A backup supplier only helps if it is working. When another disruption OVERLAPS
in time with the one being analysed, the backup's availability is reduced:

    mitigation = backup_dependency_score * (1 - impact on the backup from the
                                                 overlapping disruption(s))

(the largest such impact is used). Disruptions that do not overlap in time are
analysed as if the others did not exist. The overlap check is made against ALL
disruptions passed in, even when a window filter selects only some of them.
Only the backup's own impact is used - a backup's own backups are not chased
recursively.

Multiple disruptions are analysed one by one. A combined per-entity view is also
given, treating disruptions as independent events:
    combined = 1 - product(1 - impact_i)
Optionally restricted to the disruptions that are active at any time inside a
window (window_start / window_end). "Active at any time in the window" is a
cumulative-exposure view, not a peak at one instant. project_timeline() applies
this for 30/60/90-day horizons.

Main entry points
-----------------
analyze_disruption(graph, disruption)            -> dict for ONE disruption
analyze_disruptions(graph, disruptions)          -> dict for MANY disruptions
project_timeline(graph, disruptions, as_of)      -> combined impact per horizon
run_analysis(entities_csv, relationships_csv, disruptions_csv, output_json)
run_analysis_from_purchase_orders(po_csv, disruptions_csv, output_json)   Supplier -> SKU -> Warehouse
results_to_dataframe(analysis)                   -> one row per affected pair (for ML)
node_features_dataframe(graph, analysis)         -> one row per entity (for ML / GNN)
"""

import argparse
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd

from graph_builder import graph_summary, load_supply_chain, load_supply_chain_from_purchase_orders
from graph_utils import (
    SCHEMA_VERSION,
    SCORE_DECIMALS,
    GraphValidationError,
    check_entity_exists,
    get_flow_graph,
    risk_level,
    save_json,
)

DEFAULT_MAX_PATHS = 1000  # safety limit on how many propagation paths are listed
DEFAULT_HORIZONS = (30, 60, 90)
SAMPLE_DATA_LABEL = "SYNTHETIC SAMPLE DATA - for testing only, not real supply-chain data"

Window = Tuple[Optional[date], Optional[date]]  # (first day, last day); None = unbounded


# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------

def _parse_date(value) -> Optional[date]:
    """Accept a date, datetime or 'YYYY-MM-DD' text. Empty -> None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        raise ValueError(f"invalid date {value!r}; expected YYYY-MM-DD")


def disruption_window(disruption: Dict) -> Window:
    """(first_day, last_day) of a disruption, both inclusive.

    (None, None)  start_date unknown -> unbounded in both directions
    (start, None) duration unknown   -> open-ended, still running
    """
    start = _parse_date(disruption.get("start_date"))
    if start is None:
        return None, None
    duration = disruption.get("duration_days")
    if duration is None or duration == "" or (isinstance(duration, float) and math.isnan(duration)):
        return start, None
    days = max(1, int(math.ceil(float(duration))))
    return start, start + timedelta(days=days - 1)


def windows_overlap(a: Window, b: Window) -> bool:
    """True if two (first_day, last_day) windows share at least one day."""
    a_start, a_end = a
    b_start, b_end = b
    if a_start is not None and b_end is not None and a_start > b_end:
        return False
    if b_start is not None and a_end is not None and b_start > a_end:
        return False
    return True


def active_disruptions(disruptions: Iterable[Dict], window_start=None, window_end=None) -> List[Dict]:
    """Disruptions active on at least one day of [window_start, window_end].

    Either bound may be None (unbounded). With both None, everything is returned.
    """
    start, end = _parse_date(window_start), _parse_date(window_end)
    if start is not None and end is not None and end < start:
        raise ValueError(f"window_end {end} is before window_start {start}")
    disruptions = list(disruptions)
    if start is None and end is None:
        return disruptions
    return [d for d in disruptions if windows_overlap(disruption_window(d), (start, end))]


def _overlapping(disruption: Dict, others: Iterable[Dict]) -> List[Dict]:
    """The other disruptions that overlap this one in time (never itself)."""
    window = disruption_window(disruption)
    found = []
    for other in others:
        if other is disruption or other.get("disruption_id") == disruption.get("disruption_id"):
            continue
        if windows_overlap(window, disruption_window(other)):
            found.append(other)
    return found


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def _check_severity(severity) -> float:
    try:
        severity = float(severity)
    except (TypeError, ValueError):
        raise ValueError(f"severity must be a number between 0 and 1, got {severity!r}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be between 0 and 1, got {severity}")
    return severity


def _backup_mitigation(graph, source, target, edge, blocked: Set[str],
                       degraded: Optional[Dict[str, float]] = None) -> float:
    """Share of the lost supply that a still-working backup supplier can replace.

    degraded: entity_id -> impact (0-1) caused on it by overlapping disruptions;
    a backup that is partly hit only delivers (1 - impact) of its capacity.
    """
    material = edge.get("material")
    if not material:
        return 0.0
    degraded = degraded or {}
    best = 0.0
    for backup_source, _, backup_edge in graph.in_edges(target, data=True):
        if not backup_edge.get("is_backup", False):
            continue
        if backup_source == source or backup_source in blocked:
            continue  # the backup is the disrupted supplier, or is hit as well
        if (backup_edge.get("material") or "").lower() != material.lower():
            continue  # a backup of a different material cannot replace this one
        availability = 1.0 - degraded.get(backup_source, 0.0)
        best = max(best, backup_edge["dependency_score"] * availability)
    return best


def _concurrent_impacts(graph, overlapping: Iterable[Dict], use_ml_weights: bool = False) -> Dict[str, float]:
    """Largest impact each entity suffers from the overlapping disruptions (taken alone)."""
    degraded: Dict[str, float] = {}
    for other in overlapping:
        impacts, _ = _propagate(graph, other["entity_id"], other["severity"], use_ml_weights=use_ml_weights)
        for entity_id, score in impacts.items():
            degraded[entity_id] = max(degraded.get(entity_id, 0.0), score)
    return degraded


def _effective_dependency(edge: Dict, use_ml_weights: bool) -> float:
    """Return structural dependency_score, optionally amplified by ML risk weight."""
    base = edge["dependency_score"]
    if not use_ml_weights:
        return base
    ml_weight = float(edge.get("ml_risk_weight", 0.0) or 0.0)
    return min(1.0, base * (1.0 + max(0.0, ml_weight)))


def _propagate(graph, disrupted_id, severity,
               degraded: Optional[Dict[str, float]] = None,
               use_ml_weights: bool = False) -> Tuple[Dict[str, float], Set[Tuple[str, str]]]:
    """Compute impacts and return (impacts, edges_that_carry_impact)."""
    check_entity_exists(graph, disrupted_id)
    severity = _check_severity(severity)

    flow = get_flow_graph(graph)                  # normal flow only, no backup edges
    downstream = nx.descendants(flow, disrupted_id)
    blocked = downstream | {disrupted_id}         # entities that cannot act as backups

    impacts: Dict[str, float] = {disrupted_id: severity}
    carrying_edges: Set[Tuple[str, str]] = set()

    # Topological order guarantees a node is computed after all its affected sources.
    order = nx.topological_sort(flow.subgraph(blocked))
    for node in order:
        if node == disrupted_id:
            continue
        total = 0.0
        for source in flow.predecessors(node):
            source_impact = impacts.get(source, 0.0)
            if source_impact <= 0:
                continue
            edge = flow.edges[source, node]
            mitigation = _backup_mitigation(graph, source, node, edge, blocked, degraded)
            dep_weight = _effective_dependency(edge, use_ml_weights)
            contribution = source_impact * dep_weight * (1.0 - mitigation)
            if contribution > 0:
                total += contribution
                carrying_edges.add((source, node))
        impacts[node] = min(1.0, total)
    return impacts, carrying_edges


def calculate_impact_scores(graph, disrupted_id, severity,
                            concurrent_impacts: Optional[Dict[str, float]] = None,
                            use_ml_weights: bool = False) -> Dict[str, float]:
    """Impact score (0-1) of every entity hit by the disruption, including the
    disrupted entity itself (whose score equals the severity).

    concurrent_impacts: optional {entity_id: impact} from disruptions overlapping in
    time; it weakens the backup suppliers listed there (see module docstring).
    use_ml_weights: if True, scale edge dependency_score using attached ``ml_risk_weight``.
    """
    impacts, _ = _propagate(graph, disrupted_id, severity, concurrent_impacts, use_ml_weights=use_ml_weights)
    return impacts


def find_propagation_paths(graph, disrupted_id, severity=1.0, max_paths=DEFAULT_MAX_PATHS) -> List[List[str]]:
    """Every path along which the disruption actually travels, e.g. [['S001','F001','W001']]
    or, in a purchase-order graph, [['SUP-1','SKU-1','WH-1']]."""
    _, edges = _propagate(graph, disrupted_id, severity)
    paths, _ = _enumerate_paths(edges, disrupted_id, max_paths)
    return paths


def _enumerate_paths(edges, root, max_paths) -> Tuple[List[List[str]], bool]:
    """List all simple paths from root to each affected entity (shortest first)."""
    prop = nx.DiGraph(list(edges))
    prop.add_node(root)
    paths: List[List[str]] = []
    truncated = False
    for target in sorted(n for n in prop.nodes if n != root):
        for path in nx.all_simple_paths(prop, root, target):
            paths.append(path)
            if len(paths) > max_paths:
                truncated = True
                break
        if truncated:
            break
    paths.sort(key=lambda p: (len(p), p))
    return paths[:max_paths], truncated


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------

def analyze_disruption(graph, disruption: Dict, max_paths=DEFAULT_MAX_PATHS, min_impact=0.0,
                       concurrent_disruptions: Optional[Sequence[Dict]] = None,
                       use_ml_weights: bool = False) -> Dict:
    """Analyse one disruption and return the standardised result dictionary.

    `disruption` needs: disruption_id, entity_id, disruption_type, severity.
    Optional: location, start_date, duration_days.
    min_impact: entities with a (rounded) score <= this value are left out.
    concurrent_disruptions: other disruptions to check for time overlap. Those that
    overlap weaken the backup suppliers they hit. Leave empty to analyse in isolation.
    use_ml_weights: if True, incorporate dynamic ML risk weights on edges.
    """
    for key in ("disruption_id", "entity_id", "disruption_type", "severity"):
        if disruption.get(key) is None:
            raise ValueError(f"disruption is missing required field '{key}'")

    root = disruption["entity_id"]
    severity = _check_severity(disruption["severity"])
    start, end = disruption_window(disruption)

    overlapping = _overlapping(disruption, concurrent_disruptions or [])
    degraded = _concurrent_impacts(graph, overlapping, use_ml_weights=use_ml_weights)
    impacts, edges = _propagate(graph, root, severity, degraded, use_ml_weights=use_ml_weights)
    paths, truncated = _enumerate_paths(edges, root, max_paths)

    flow = nx.DiGraph(list(edges))
    flow.add_node(root)

    affected = []
    for entity_id, score in impacts.items():
        score = round(score, SCORE_DECIMALS)
        if entity_id == root or score <= min_impact:
            continue
        node = graph.nodes[entity_id]
        affected.append({
            "entity_id": entity_id,
            "entity_name": node.get("entity_name"),
            "entity_type": node["entity_type"],
            "location": node.get("location"),
            "impact_score": score,
            "risk_level": risk_level(score),
            "hops": nx.shortest_path_length(flow, root, entity_id),
        })
    affected.sort(key=lambda a: (-a["impact_score"], a["entity_id"]))

    by_type: Dict[str, int] = {}
    for item in affected:
        by_type[item["entity_type"]] = by_type.get(item["entity_type"], 0) + 1

    return {
        "disruption_id": disruption["disruption_id"],
        "disrupted_entity_id": root,
        "disrupted_entity_type": graph.nodes[root]["entity_type"],
        "disruption_type": disruption["disruption_type"],
        "severity": severity,
        "location": disruption.get("location"),
        "start_date": start.isoformat() if start else None,
        "end_date": end.isoformat() if end else None,   # None = open-ended or unknown start
        "duration_days": disruption.get("duration_days"),
        "concurrent_disruption_ids": sorted(str(o["disruption_id"]) for o in overlapping),
        "affected_entities": affected,
        "propagation_paths": paths,
        "paths_truncated": truncated,
        "summary": {
            "total_affected": len(affected),
            "affected_by_type": by_type,
            "highest_risk_level": affected[0]["risk_level"] if affected else "none",
        },
    }


def combine_impacts(graph, disruptions: List[Dict], window_start=None, window_end=None,
                    use_ml_weights: bool = False) -> List[Dict]:
    """Per-entity combined impact over several disruptions (independent events).

    window_start / window_end: only disruptions active on at least one day of this
    window are combined. Overlap (for backup availability) is still checked against
    ALL the disruptions passed in.
    """
    disruptions = list(disruptions)
    selected = active_disruptions(disruptions, window_start, window_end)

    survival: Dict[str, float] = {}   # probability-like "still fine" factor
    sources: Dict[str, List[str]] = {}
    for d in selected:
        degraded = _concurrent_impacts(graph, _overlapping(d, disruptions), use_ml_weights=use_ml_weights)
        impacts = calculate_impact_scores(graph, d["entity_id"], d["severity"], degraded,
                                          use_ml_weights=use_ml_weights)
        for entity_id, score in impacts.items():
            if score > 0:
                survival[entity_id] = survival.get(entity_id, 1.0) * (1.0 - score)
                sources.setdefault(entity_id, []).append(d["disruption_id"])

    directly_hit = {d["entity_id"] for d in selected}
    combined = []
    for entity_id, factor in survival.items():
        score = round(1.0 - factor, SCORE_DECIMALS)
        node = graph.nodes[entity_id]
        combined.append({
            "entity_id": entity_id,
            "entity_name": node.get("entity_name"),
            "entity_type": node["entity_type"],
            "combined_impact_score": score,
            "risk_level": risk_level(score),
            "directly_disrupted": entity_id in directly_hit,
            "disruption_ids": sources[entity_id],
        })
    combined.sort(key=lambda c: (-c["combined_impact_score"], c["entity_id"]))
    return combined


def analyze_disruptions(graph, disruptions: List[Dict], max_paths=DEFAULT_MAX_PATHS,
                        min_impact=0.0, data_label: Optional[str] = None,
                        window_start=None, window_end=None,
                        use_ml_weights: bool = False) -> Dict:
    """Analyse many disruptions. Returns {'results': [...], 'combined_impact': [...], ...}.

    window_start / window_end (optional): restrict results and the combined view to
    disruptions active on at least one day of the window.
    """
    disruptions = list(disruptions)
    selected = active_disruptions(disruptions, window_start, window_end)
    results = [analyze_disruption(graph, d, max_paths, min_impact,
                                  concurrent_disruptions=disruptions,
                                  use_ml_weights=use_ml_weights)
               for d in selected]
    start, end = _parse_date(window_start), _parse_date(window_end)
    return {
        "schema_version": SCHEMA_VERSION,
        "data_label": data_label,
        "time_window": {"start": start.isoformat() if start else None,
                        "end": end.isoformat() if end else None},
        "disruption_count": len(results),
        "results": results,
        "combined_impact": combine_impacts(graph, disruptions, window_start, window_end,
                                           use_ml_weights=use_ml_weights),
    }


def project_timeline(graph, disruptions: List[Dict], as_of,
                     horizons: Sequence[int] = DEFAULT_HORIZONS) -> List[Dict]:
    """Combined impact for several look-ahead horizons (default 30/60/90 days).

    Horizon h covers as_of .. as_of + h - 1 (inclusive). Each entry lists the
    disruptions that are active at some point in that window and their combined impact.
    Disruptions that ended before `as_of` are left out; open-ended ones always count.
    """
    start = _parse_date(as_of)
    if start is None:
        raise ValueError("as_of is required (YYYY-MM-DD)")
    disruptions = list(disruptions)
    timeline = []
    for horizon in horizons:
        if int(horizon) != horizon or horizon < 1:
            raise ValueError(f"horizon must be a positive whole number of days, got {horizon!r}")
        end = start + timedelta(days=int(horizon) - 1)
        timeline.append({
            "horizon_days": int(horizon),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "disruption_ids": [d["disruption_id"] for d in active_disruptions(disruptions, start, end)],
            "combined_impact": combine_impacts(graph, disruptions, start, end),
        })
    return timeline


# ---------------------------------------------------------------------------
# Tables for ML
# ---------------------------------------------------------------------------

def results_to_dataframe(analysis: Dict) -> pd.DataFrame:
    """Flatten the analysis into one row per (disruption, affected entity).

    Handy as ML input. Accepts the dict from analyze_disruptions, or a list of
    single-disruption results. Only AFFECTED entities appear - use
    node_features_dataframe() for a row per entity including unaffected ones.
    """
    results = analysis["results"] if isinstance(analysis, dict) and "results" in analysis else analysis
    rows = []
    for result in results:
        for entity in result["affected_entities"]:
            rows.append({
                "disruption_id": result["disruption_id"],
                "disrupted_entity_id": result["disrupted_entity_id"],
                "disruption_type": result["disruption_type"],
                "severity": result["severity"],
                "start_date": result["start_date"],
                "duration_days": result["duration_days"],
                "affected_entity_id": entity["entity_id"],
                "affected_entity_type": entity["entity_type"],
                "impact_score": entity["impact_score"],
                "risk_level": entity["risk_level"],
                "hops": entity["hops"],
            })
    return pd.DataFrame(rows)


def structural_features(graph: nx.DiGraph) -> Dict[str, Dict]:
    """Per-entity structural features (no disruption needed). entity_id -> feature dict.

    in_degree / out_degree      count NORMAL flow edges only (backup edges excluded)
    backup_in_degree            backup suppliers feeding this entity
    backup_out_degree           entities this one is a backup for
    upstream_count / downstream_count   all ancestors / descendants in the normal flow
    total_dependency_in         sum of normal inbound dependency_scores
    max_dependency_in           largest single normal inbound dependency_score
    has_backup                  1 if at least one backup supplier feeds this entity, else 0
    backup_coverage             share (0-1) of the normal inbound dependency whose material
                                has at least one backup supplier (0 if no inbound edges)
    """
    flow = get_flow_graph(graph)
    features: Dict[str, Dict] = {}
    for node, data in graph.nodes(data=True):
        normal_in = [d for _, _, d in graph.in_edges(node, data=True) if not d.get("is_backup", False)]
        backup_in = [d for _, _, d in graph.in_edges(node, data=True) if d.get("is_backup", False)]
        backup_out = [1 for _, _, d in graph.out_edges(node, data=True) if d.get("is_backup", False)]

        backup_materials = {(d.get("material") or "").lower() for d in backup_in if d.get("material")}
        total_in = sum(d["dependency_score"] for d in normal_in)
        covered = sum(d["dependency_score"] for d in normal_in
                      if d.get("material") and d["material"].lower() in backup_materials)

        features[node] = {
            "entity_id": node,
            "entity_name": data.get("entity_name"),
            "entity_type": data.get("entity_type"),
            "location": data.get("location"),
            "in_degree": flow.in_degree(node),
            "out_degree": flow.out_degree(node),
            "backup_in_degree": len(backup_in),
            "backup_out_degree": len(backup_out),
            "upstream_count": len(nx.ancestors(flow, node)),
            "downstream_count": len(nx.descendants(flow, node)),
            "total_dependency_in": round(total_in, SCORE_DECIMALS),
            "max_dependency_in": round(max((d["dependency_score"] for d in normal_in), default=0.0),
                                       SCORE_DECIMALS),
            "has_backup": int(len(backup_in) > 0),
            "backup_coverage": round(covered / total_in, SCORE_DECIMALS) if total_in > 0 else 0.0,
        }
    return features


def node_features_dataframe(graph: nx.DiGraph, analysis: Optional[Dict] = None,
                            per_disruption: bool = False) -> pd.DataFrame:
    """One row for EVERY entity in the graph, including unaffected ones (impact 0).

    Always included: the structural columns of structural_features().

    analysis=None: structural columns only.

    per_disruption=False (default) - one row per entity, summarising all disruptions
    in `analysis`:
        combined_impact_score   combined score over the analysed disruptions (0 if untouched)
        risk_level              'high' / 'medium' / 'low' / 'none'
        max_single_impact       largest impact from any single disruption (0 if untouched)
        n_disruptions           how many disruptions touch this entity
        directly_disrupted      1 if a disruption targets this entity itself
        min_hops                fewest hops from any disruption (0 = directly hit,
                                NaN = never reached)

    per_disruption=True - one row per (disruption, entity), i.e. every entity for every
    disruption, with that disruption's attributes plus:
        is_disrupted_entity, impact_score (0 if unaffected), risk_level, hops (NaN if
        unaffected), n_concurrent (disruptions overlapping in time).

    NOTE: impact_score is itself computed from the dependency formula, not observed.
    A model trained to predict it from these structural features mostly learns the
    formula; observed delays or losses are the real target for delay prediction.
    """
    structure = structural_features(graph)
    entity_ids = sorted(structure)

    if per_disruption:
        if analysis is None:
            raise ValueError("per_disruption=True needs an analysis dict")
        rows = []
        for result in analysis["results"]:
            root = result["disrupted_entity_id"]
            affected = {a["entity_id"]: a for a in result["affected_entities"]}
            for entity_id in entity_ids:
                hit = affected.get(entity_id)
                if entity_id == root:
                    score, level, hops = result["severity"], risk_level(result["severity"]), 0
                elif hit:
                    score, level, hops = hit["impact_score"], hit["risk_level"], hit["hops"]
                else:
                    score, level, hops = 0.0, "none", float("nan")
                rows.append({
                    "disruption_id": result["disruption_id"],
                    "disrupted_entity_id": root,
                    "disruption_type": result["disruption_type"],
                    "severity": result["severity"],
                    "start_date": result["start_date"],
                    "duration_days": result["duration_days"],
                    "n_concurrent": len(result.get("concurrent_disruption_ids", [])),
                    **structure[entity_id],
                    "is_disrupted_entity": int(entity_id == root),
                    "impact_score": score,
                    "risk_level": level,
                    "hops": hops,
                })
        return pd.DataFrame(rows)

    rows = []
    combined = {c["entity_id"]: c for c in analysis["combined_impact"]} if analysis else {}
    max_single: Dict[str, float] = {}
    min_hops: Dict[str, int] = {}
    for result in (analysis["results"] if analysis else []):
        root = result["disrupted_entity_id"]
        max_single[root] = max(max_single.get(root, 0.0), result["severity"])
        min_hops[root] = 0
        for a in result["affected_entities"]:
            max_single[a["entity_id"]] = max(max_single.get(a["entity_id"], 0.0), a["impact_score"])
            min_hops[a["entity_id"]] = min(min_hops.get(a["entity_id"], a["hops"]), a["hops"])

    for entity_id in entity_ids:
        row = dict(structure[entity_id])
        if analysis is not None:
            c = combined.get(entity_id)
            row.update({
                "combined_impact_score": c["combined_impact_score"] if c else 0.0,
                "risk_level": c["risk_level"] if c else "none",
                "max_single_impact": max_single.get(entity_id, 0.0),
                "n_disruptions": len(c["disruption_ids"]) if c else 0,
                "directly_disrupted": int(bool(c and c["directly_disrupted"])),
                "min_hops": min_hops.get(entity_id, float("nan")),
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dynamic ML risk weighting & inventory ripple estimation
# ---------------------------------------------------------------------------

def apply_ml_risk_weights(graph: nx.DiGraph,
                          node_risks: Optional[Dict[str, float]] = None,
                          edge_risks: Optional[Dict[Tuple[str, str], float]] = None,
                          risk_blend: float = 0.5) -> nx.DiGraph:
    """Return a copy of ``graph`` annotated with ML-predicted node/edge risk weights.

    node_risks: {entity_id: predicted_disruption_probability (0..1)}
    edge_risks: {(source_id, target_id): predicted_delay_probability (0..1)}
    risk_blend: multiplier (default 0.5) controlling how strongly ML risk amplifies
                structural dependency_score when ``use_ml_weights=True``.
    """
    weighted = graph.copy()
    node_risks = node_risks or {}
    edge_risks = edge_risks or {}

    for node, data in weighted.nodes(data=True):
        prob = float(node_risks.get(node, 0.0) or 0.0)
        data["ml_risk_prob"] = round(max(0.0, min(1.0, prob)), SCORE_DECIMALS)

    for u, v, data in weighted.edges(data=True):
        if (u, v) in edge_risks:
            edge_prob = float(edge_risks[(u, v)] or 0.0)
        else:
            edge_prob = max(float(node_risks.get(u, 0.0) or 0.0), float(node_risks.get(v, 0.0) or 0.0))
        edge_prob = max(0.0, min(1.0, edge_prob))
        data["ml_edge_risk"] = round(edge_prob, SCORE_DECIMALS)
        data["ml_risk_weight"] = round(max(0.0, risk_blend) * edge_prob, SCORE_DECIMALS)
    return weighted


def generate_ml_risk_disruptions(graph: nx.DiGraph,
                                 node_risks: Dict[str, float],
                                 threshold: float = 0.40,
                                 expected_delays: Optional[Dict[str, float]] = None,
                                 entity_types: Optional[Sequence[str]] = ("supplier", "sku"),
                                 as_of_date: Optional[str] = None) -> List[Dict]:
    """Convert high-risk ML node predictions into disruption dictionaries for propagation."""
    expected_delays = expected_delays or {}
    allowed_types = set(entity_types) if entity_types is not None else None
    events = []
    idx = 1
    for entity_id, prob in sorted(node_risks.items(), key=lambda kv: (-kv[1], kv[0])):
        if entity_id not in graph:
            continue
        etype = graph.nodes[entity_id].get("entity_type")
        if allowed_types is not None and etype not in allowed_types:
            continue
        prob = float(prob)
        if prob < threshold:
            continue
        delay = expected_delays.get(entity_id)
        events.append({
            "disruption_id": f"ML-RISK-{idx:03d}",
            "entity_id": entity_id,
            "disruption_type": "ml_predicted_delay",
            "severity": round(max(0.0, min(1.0, prob)), SCORE_DECIMALS),
            "location": graph.nodes[entity_id].get("location"),
            "start_date": as_of_date,
            "duration_days": max(1, int(round(delay))) if delay is not None else 7,
        })
        idx += 1
    return events


def estimate_inventory_ripple(analysis_result: Dict,
                              inventory_df: pd.DataFrame,
                              simulated_delay_days: float = 7.0) -> pd.DataFrame:
    """Estimate downstream SKU stockout exposure from a disruption analysis result.

    Compares effective delay days (simulated_delay_days * impact_score) against
    each affected SKU's ``days_of_stock``, ``current_stock_qty``, and ``safety_stock_qty``.
    """
    if inventory_df.empty:
        return pd.DataFrame()
    inv_idx = inventory_df.drop_duplicates("sku_id").set_index("sku_id")
    rows = []

    candidates = list(analysis_result.get("affected_entities", []))
    root_id = analysis_result.get("disrupted_entity_id")
    if root_id and analysis_result.get("disrupted_entity_type") == "sku":
        candidates.insert(0, {
            "entity_id": root_id,
            "entity_type": "sku",
            "impact_score": analysis_result.get("severity", 1.0),
            "risk_level": risk_level(analysis_result.get("severity", 1.0)),
            "hops": 0,
        })

    for item in candidates:
        if item.get("entity_type") != "sku":
            continue
        sku_id = item["entity_id"]
        if sku_id not in inv_idx.index:
            continue
        inv = inv_idx.loc[sku_id]
        impact = float(item["impact_score"])
        effective_delay = round(float(simulated_delay_days) * impact, 2)
        dos = float(inv.get("days_of_stock", 0.0) or 0.0)
        daily_demand = float(inv.get("avg_weekly_demand", 0.0) or 0.0) / 7.0
        projected_units_at_risk = round(daily_demand * effective_delay, 1)
        remaining_dos = round(dos - effective_delay, 1)
        curr_stock = float(inv.get("current_stock_qty", 0.0) or 0.0)
        safety_stock = float(inv.get("safety_stock_qty", 0.0) or 0.0)
        post_shock_stock = round(max(0.0, curr_stock - projected_units_at_risk), 1)
        stockout_risk = "Stockout" if remaining_dos <= 0 else (
            "Safety Breach" if post_shock_stock < safety_stock else "Buffered"
        )
        rows.append({
            "sku_id": sku_id,
            "sku_name": inv.get("sku_name", sku_id),
            "warehouse": inv.get("warehouse", ""),
            "abc_class": inv.get("abc_class", ""),
            "impact_score": impact,
            "risk_level": item["risk_level"],
            "hops": item["hops"],
            "current_stock_qty": int(curr_stock),
            "safety_stock_qty": int(safety_stock),
            "days_of_stock": round(dos, 1),
            "effective_delay_days": effective_delay,
            "projected_units_at_risk": projected_units_at_risk,
            "post_shock_stock_qty": post_shock_stock,
            "remaining_buffer_days": remaining_dos,
            "stockout_status": stockout_risk,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["impact_score", "remaining_buffer_days"], ascending=[False, True]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def run_analysis(entities_csv, relationships_csv, disruptions_csv, output_json=None,
                 strict=True, data_label=None, max_paths=DEFAULT_MAX_PATHS,
                 as_of=None, horizons: Sequence[int] = DEFAULT_HORIZONS) -> Dict:
    """End to end: CSV files -> graph -> analysis (-> JSON file). Returns the analysis dict.

    as_of (optional, 'YYYY-MM-DD'): also add a 'timeline' with combined impact for each
    look-ahead horizon (default 30/60/90 days).
    """
    data = load_supply_chain(entities_csv, relationships_csv, disruptions_csv, strict=strict)
    return _analyse_loaded(data, output_json, data_label, max_paths, as_of, horizons)


def run_analysis_from_purchase_orders(purchase_orders_csv, disruptions_csv, output_json=None,
                                      strict=True, data_label=None, max_paths=DEFAULT_MAX_PATHS,
                                      as_of=None, horizons: Sequence[int] = DEFAULT_HORIZONS,
                                      columns=None, backup_pairs=None) -> Dict:
    """Like run_analysis, but the Supplier -> SKU -> Warehouse graph is built from a
    purchase-order CSV (see graph_builder.load_supply_chain_from_purchase_orders)."""
    data = load_supply_chain_from_purchase_orders(purchase_orders_csv, disruptions_csv, columns,
                                                  backup_pairs, strict=strict)
    return _analyse_loaded(data, output_json, data_label, max_paths, as_of, horizons)


def _analyse_loaded(data, output_json, data_label, max_paths, as_of, horizons) -> Dict:
    analysis = analyze_disruptions(data.graph, data.disruptions, max_paths=max_paths, data_label=data_label)
    if as_of is not None:
        analysis["timeline"] = project_timeline(data.graph, data.disruptions, as_of, horizons)
    analysis["graph_summary"] = graph_summary(data.graph)
    analysis["validation"] = data.report.to_dict()
    if output_json:
        save_json(analysis, output_json)
    return analysis


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run disruption impact analysis.")
    parser.add_argument("--data-dir", default=str(here / "data"),
                        help="folder with entities.csv, relationships.csv, disruptions.csv")
    parser.add_argument("--output", default=str(here / "sample_impact_results.json"),
                        help="where to write the JSON results")
    parser.add_argument("--purchase-orders", help="purchase-order CSV: build the Supplier -> SKU -> "
                                                  "Warehouse graph from it (disruptions.csv still comes "
                                                  "from --data-dir)")
    parser.add_argument("--label", help="text describing the data source, stored in the JSON")
    parser.add_argument("--lenient", action="store_true", help="drop invalid rows instead of failing")
    parser.add_argument("--as-of", help="YYYY-MM-DD: add 30/60/90-day timeline projections from this date")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS),
                        help="look-ahead horizons in days (used with --as-of)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    using_sample = data_dir.resolve() == (here / "data").resolve()
    label = args.label or (SAMPLE_DATA_LABEL if using_sample else "user-supplied dataset")

    try:
        if args.purchase_orders:
            analysis = run_analysis_from_purchase_orders(
                args.purchase_orders, data_dir / "disruptions.csv", args.output,
                strict=not args.lenient, data_label=label, as_of=args.as_of, horizons=args.horizons)
        else:
            analysis = run_analysis(data_dir / "entities.csv", data_dir / "relationships.csv",
                                    data_dir / "disruptions.csv", args.output,
                                    strict=not args.lenient, data_label=label,
                                    as_of=args.as_of, horizons=args.horizons)
    except (FileNotFoundError, GraphValidationError, ValueError) as error:
        raise SystemExit(f"Analysis failed:\n{error}")

    print(f"Data: {label}")
    for result in analysis["results"]:
        print(f"\n{result['disruption_id']}: {result['disruption_type']} at "
              f"{result['disrupted_entity_id']} (severity {result['severity']})")
        if result["concurrent_disruption_ids"]:
            print(f"  overlaps in time with: {', '.join(result['concurrent_disruption_ids'])}")
        if not result["affected_entities"]:
            print("  no downstream entities affected")
        for e in result["affected_entities"]:
            print(f"  {e['entity_id']} ({e['entity_type']}): impact {e['impact_score']} [{e['risk_level']}]")
    for step in analysis.get("timeline", []):
        worst = step["combined_impact"][0] if step["combined_impact"] else None
        print(f"\n{step['horizon_days']}-day window {step['window_start']} .. {step['window_end']}: "
              f"{len(step['disruption_ids'])} active disruption(s)"
              + (f", highest {worst['entity_id']} {worst['combined_impact_score']}" if worst else ""))
    print(f"\nJSON written to {args.output}")


if __name__ == "__main__":
    main()
