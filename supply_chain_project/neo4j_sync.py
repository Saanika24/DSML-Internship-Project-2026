"""
neo4j_sync.py
=============
Cypher export / sync helpers for loading the supply-chain graph into Neo4j and
keeping node risk states up to date from live disruption events (for example,
events produced by the NLP pipeline).

Nothing here imports the `neo4j` package. Every function that builds Cypher
returns plain (query, parameters) pairs, so the output can be inspected, tested
and exported without a database. Only run_statements() / sync_graph() /
sync_risk_state() / apply_event() talk to a database, and only when you pass in
a driver (from `neo4j.GraphDatabase.driver(...)`, neo4j >= 5).

All values travel as query PARAMETERS. Only labels and relationship types are
written into the query text (Cypher cannot parameterise them), and those are
checked against a strict identifier pattern first.

Graph model
-----------
(:Entity:Supplier|Factory|Sku|Warehouse {entity_id, entity_name, location, ...,
                                      risk_score, risk_level, directly_disrupted,
                                      active_disruption_ids, risk_updated_at})
(a)-[:SUPPLIES|SHIPS_TO {dependency_score, material, is_backup}]->(b)
(:Disruption {disruption_id, disruption_type, severity, location, start_date,
              end_date, duration_days})
(:Disruption)-[:DISRUPTS]->(:Entity)                    the entity hit directly
(:Disruption)-[:IMPACTS {impact_score, risk_level, hops}]->(:Entity)   knock-on effects

Dates are stored as ISO text (YYYY-MM-DD). Re-running a sync is safe: nodes and
relationships are MERGEd, and live risk-state properties are only initialised
when a node is first created.

Main entry points
-----------------
graph_to_cypher(graph)                      -> statements that load the graph
risk_state_statements(graph, combined)      -> statements that set each node's risk state
apply_event(graph, event, active_events)    -> single live event: persist it and refresh risk states
sync_risk_state(graph, active_events)       -> recompute risk states from a set of active events
sync_graph(graph, driver)                   -> load the graph into a database
export_statements(statements, path)         -> save statements as JSON for review or other tools
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from graph_builder import load_supply_chain
from graph_utils import GraphValidationError, save_json
from impact_analysis import analyze_disruption, combine_impacts

Statement = Tuple[str, Dict[str, Any]]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SCHEMA_STATEMENTS: List[Statement] = [
    ("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
     "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE", {}),
    ("CREATE CONSTRAINT disruption_id_unique IF NOT EXISTS "
     "FOR (d:Disruption) REQUIRE d.disruption_id IS UNIQUE", {}),
]


def _safe_identifier(value: str) -> str:
    """Labels / relationship types go into query text, so they must be plain identifiers."""
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ValueError(f"unsafe Cypher identifier: {value!r}")
    return value


def _label(entity_type: str) -> str:
    return _safe_identifier(entity_type.capitalize())        # supplier -> Supplier


def _rel_type(relationship_type: str) -> str:
    return _safe_identifier(relationship_type.upper())       # ships_to -> SHIPS_TO


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 1. Load the graph
# ---------------------------------------------------------------------------

def schema_statements() -> List[Statement]:
    """Uniqueness constraints on entity_id and disruption_id (idempotent)."""
    return [(q, dict(p)) for q, p in SCHEMA_STATEMENTS]


def graph_to_cypher(graph: nx.DiGraph, include_schema: bool = True) -> List[Statement]:
    """(query, parameters) pairs that create or update the whole graph in Neo4j.

    Order: constraints, entity nodes (one statement per entity type), relationships
    (one statement per relationship type). Nodes get risk_score 0.0 / risk_level
    'none' only on first creation, so a re-sync never wipes live risk states.
    """
    statements: List[Statement] = schema_statements() if include_schema else []

    by_type: Dict[str, List[Dict]] = {}
    for node, data in graph.nodes(data=True):
        props = {k: v for k, v in data.items() if k != "entity_id"}
        by_type.setdefault(data["entity_type"], []).append({"entity_id": node, "props": props})
    for entity_type in sorted(by_type):
        query = (
            "UNWIND $rows AS row\n"
            "MERGE (e:Entity {entity_id: row.entity_id})\n"
            "ON CREATE SET e.risk_score = 0.0, e.risk_level = 'none', "
            "e.directly_disrupted = false, e.active_disruption_ids = []\n"
            f"SET e:{_label(entity_type)}\n"
            "SET e += row.props"
        )
        statements.append((query, {"rows": sorted(by_type[entity_type], key=lambda r: r["entity_id"])}))

    by_rel: Dict[str, List[Dict]] = {}
    for u, v, data in graph.edges(data=True):
        props = {k: val for k, val in data.items()}
        by_rel.setdefault(data["relationship_type"], []).append(
            {"source_id": u, "target_id": v, "props": props})
    for rel_type in sorted(by_rel):
        query = (
            "UNWIND $rows AS row\n"
            "MATCH (a:Entity {entity_id: row.source_id})\n"
            "MATCH (b:Entity {entity_id: row.target_id})\n"
            f"MERGE (a)-[r:{_rel_type(rel_type)}]->(b)\n"
            "SET r += row.props"
        )
        rows = sorted(by_rel[rel_type], key=lambda r: (r["source_id"], r["target_id"]))
        statements.append((query, {"rows": rows}))
    return statements


# ---------------------------------------------------------------------------
# 2. Risk states and events
# ---------------------------------------------------------------------------

def risk_state_statements(graph: nx.DiGraph, combined_impact: List[Dict],
                          updated_at: Optional[str] = None) -> List[Statement]:
    """Set risk_score / risk_level / directly_disrupted / active_disruption_ids on EVERY
    entity. Entities missing from `combined_impact` are reset to 0 / 'none', so states
    from disruptions that have ended are cleared.
    """
    updated_at = updated_at or _now()
    by_id = {c["entity_id"]: c for c in combined_impact}
    rows = []
    for entity_id in sorted(graph.nodes):
        c = by_id.get(entity_id)
        rows.append({
            "entity_id": entity_id,
            "risk_score": c["combined_impact_score"] if c else 0.0,
            "risk_level": c["risk_level"] if c else "none",
            "directly_disrupted": bool(c and c["directly_disrupted"]),
            "disruption_ids": list(c["disruption_ids"]) if c else [],
            "updated_at": updated_at,
        })
    query = (
        "UNWIND $rows AS row\n"
        "MATCH (e:Entity {entity_id: row.entity_id})\n"
        "SET e.risk_score = row.risk_score,\n"
        "    e.risk_level = row.risk_level,\n"
        "    e.directly_disrupted = row.directly_disrupted,\n"
        "    e.active_disruption_ids = row.disruption_ids,\n"
        "    e.risk_updated_at = row.updated_at"
    )
    return [(query, {"rows": rows})]


def disruption_statements(result: Dict) -> List[Statement]:
    """Statements that store ONE analysed disruption (output of analyze_disruption):
    the :Disruption node, its DISRUPTS edge and its IMPACTS edges. Old edges of the
    same disruption_id are removed first, so re-sending an updated event replaces it.
    """
    disruption_id = result["disruption_id"]
    props = {
        "disruption_type": result["disruption_type"],
        "severity": result["severity"],
        "location": result.get("location"),
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date"),
        "duration_days": result.get("duration_days"),
    }
    impacts = [{"entity_id": a["entity_id"], "impact_score": a["impact_score"],
                "risk_level": a["risk_level"], "hops": a["hops"]}
               for a in result["affected_entities"]]
    return [
        ("MERGE (d:Disruption {disruption_id: $disruption_id})\nSET d += $props",
         {"disruption_id": disruption_id, "props": props}),
        ("MATCH (:Disruption {disruption_id: $disruption_id})-[r:DISRUPTS|IMPACTS]->()\nDELETE r",
         {"disruption_id": disruption_id}),
        ("MATCH (d:Disruption {disruption_id: $disruption_id})\n"
         "MATCH (e:Entity {entity_id: $entity_id})\n"
         "MERGE (d)-[:DISRUPTS]->(e)",
         {"disruption_id": disruption_id, "entity_id": result["disrupted_entity_id"]}),
        ("UNWIND $rows AS row\n"
         "MATCH (d:Disruption {disruption_id: $disruption_id})\n"
         "MATCH (e:Entity {entity_id: row.entity_id})\n"
         "MERGE (d)-[r:IMPACTS]->(e)\n"
         "SET r.impact_score = row.impact_score, r.risk_level = row.risk_level, r.hops = row.hops",
         {"disruption_id": disruption_id, "rows": impacts}),
    ]


def sync_risk_state(graph: nx.DiGraph, active_events: Sequence[Dict], as_of=None,
                    driver=None, database: Optional[str] = None,
                    updated_at: Optional[str] = None) -> Dict:
    """Recompute every entity's risk state from a set of events and (optionally) write it.

    as_of: only events whose time window contains this date count (None = all of them).
    To clear an ended event, drop it from `active_events` and call this again.
    Returns {'combined_impact': [...], 'statements': [...]}.
    """
    events = list(active_events)
    combined = combine_impacts(graph, events, as_of, as_of)
    statements = risk_state_statements(graph, combined, updated_at)
    if driver is not None:
        run_statements(driver, statements, database)
    return {"combined_impact": combined, "statements": statements}


def apply_event(graph: nx.DiGraph, event: Dict, active_events: Optional[Sequence[Dict]] = None,
                as_of=None, driver=None, database: Optional[str] = None,
                updated_at: Optional[str] = None) -> Dict:
    """Handle ONE live disruption event (e.g. from the NLP pipeline).

    event: needs disruption_id, entity_id, disruption_type, severity; optional location,
    start_date (YYYY-MM-DD), duration_days. An event with an existing disruption_id
    replaces the earlier version.

    Steps: validate and analyse the event (overlap with `active_events` weakens backup
    suppliers), store it as a :Disruption, then refresh the risk state of every entity
    from all active events together. Bad input raises ValueError / EntityNotFoundError
    BEFORE anything is written.

    active_events: the events already in force (the caller keeps this list; pass back
    the returned 'active_events' next time). as_of: count only events active on that date.
    driver: a Neo4j driver; None = only build the statements (dry run).

    Returns {'event_result', 'combined_impact', 'active_events', 'statements'}.
    """
    others = [e for e in (active_events or []) if e.get("disruption_id") != event.get("disruption_id")]
    all_events = others + [event]

    event_result = analyze_disruption(graph, event, concurrent_disruptions=all_events)  # validates
    combined = combine_impacts(graph, all_events, as_of, as_of)

    updated_at = updated_at or _now()
    statements = disruption_statements(event_result) + risk_state_statements(graph, combined, updated_at)
    if driver is not None:
        run_statements(driver, statements, database)
    return {"event_result": event_result, "combined_impact": combined,
            "active_events": all_events, "statements": statements}


# ---------------------------------------------------------------------------
# 3. Running statements
# ---------------------------------------------------------------------------

def _is_schema(query: str) -> bool:
    return query.lstrip().upper().startswith(("CREATE CONSTRAINT", "CREATE INDEX", "DROP CONSTRAINT"))


def run_statements(driver, statements: Sequence[Statement], database: Optional[str] = None) -> None:
    """Run statements against Neo4j (driver from neo4j >= 5).

    Neo4j does not allow schema changes and data writes in one transaction, so schema
    statements run first on their own, then all other statements run together in ONE
    write transaction (all succeed or none are applied).
    """
    schema = [s for s in statements if _is_schema(s[0])]
    data = [s for s in statements if not _is_schema(s[0])]
    with driver.session(database=database) as session:
        for query, params in schema:
            session.run(query, params)
        if data:
            def work(tx):
                for query, params in data:
                    tx.run(query, params)
            session.execute_write(work)


def sync_graph(graph: nx.DiGraph, driver, database: Optional[str] = None) -> List[Statement]:
    """Load (or update) the whole graph in Neo4j. Returns the statements that were run."""
    statements = graph_to_cypher(graph)
    run_statements(driver, statements, database)
    return statements


def export_statements(statements: Sequence[Statement], path) -> Path:
    """Write statements as JSON [{'query': ..., 'parameters': {...}}, ...] for review."""
    return save_json([{"query": q, "parameters": p} for q, p in statements], path)


# ---------------------------------------------------------------------------
# Command line (export only - never connects to a database)
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export the Cypher statements that load the graph.")
    parser.add_argument("--data-dir", default=str(here / "data"),
                        help="folder with entities.csv, relationships.csv, disruptions.csv")
    parser.add_argument("--output", default=str(here / "cypher_statements.json"),
                        help="where to write the statements (JSON)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    try:
        data = load_supply_chain(data_dir / "entities.csv", data_dir / "relationships.csv",
                                 data_dir / "disruptions.csv")
    except (FileNotFoundError, GraphValidationError) as error:
        raise SystemExit(f"Could not build the graph:\n{error}")
    statements = graph_to_cypher(data.graph)
    print(f"{len(statements)} statements written to {export_statements(statements, args.output)}")


if __name__ == "__main__":
    main()
