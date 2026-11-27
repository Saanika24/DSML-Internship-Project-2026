"""
Unit tests for the supply-chain modules.

Run from the project folder with either:
    python -m unittest -v
    pytest

All tests use the bundled SYNTHETIC sample data in ./data or small inline fixtures.
No database is needed: the Neo4j tests use a fake driver.
"""

import copy
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

import graph_utils
import impact_analysis as ia
import neo4j_sync as ns
from graph_builder import (build_supply_chain, build_supply_chain_from_purchase_orders,
                           graph_summary, load_supply_chain, load_supply_chain_from_purchase_orders)
from graph_utils import EntityNotFoundError, GraphValidationError

DATA_DIR = Path(__file__).resolve().parent / "data"
PO_DIR = DATA_DIR / "po_sample"          # synthetic purchase-order sample (Supplier -> SKU -> Warehouse)


def sample():
    return load_supply_chain(DATA_DIR / "entities.csv", DATA_DIR / "relationships.csv",
                             DATA_DIR / "disruptions.csv")


def po_sample():
    return load_supply_chain_from_purchase_orders(PO_DIR / "purchase_orders.csv", PO_DIR / "disruptions.csv")


def po_frame(rows):
    """Small purchase-order DataFrame: rows of (supplier_id, sku_id, warehouse_id, quantity[, is_backup])."""
    cols = ["supplier_id", "sku_id", "warehouse_id", "quantity", "is_backup"]
    return pd.DataFrame([list(r) + ["false"] * (5 - len(r)) for r in rows], columns=cols)


def frames(entities, relationships, disruptions=None):
    cols_e = ["entity_id", "entity_name", "entity_type", "location"]
    cols_r = ["source_id", "target_id", "relationship_type", "dependency_score", "material", "is_backup"]
    cols_d = ["disruption_id", "entity_id", "disruption_type", "severity", "location",
              "start_date", "duration_days"]
    return (pd.DataFrame(entities, columns=cols_e), pd.DataFrame(relationships, columns=cols_r),
            None if disruptions is None else pd.DataFrame(disruptions, columns=cols_d))


def disruption(did, entity, severity, start=None, days=None, kind="test"):
    return {"disruption_id": did, "entity_id": entity, "disruption_type": kind, "severity": severity,
            "location": None, "start_date": start, "duration_days": days}


# ---------------------------------------------------------------------------
# Graph construction and validation
# ---------------------------------------------------------------------------

class GraphBuilderTests(unittest.TestCase):
    def test_sample_data_loads_cleanly(self):
        data = sample()
        self.assertTrue(data.report.is_valid)
        self.assertEqual(len(data.report.issues), 0)
        self.assertEqual(data.graph.number_of_nodes(), 12)
        self.assertEqual(data.graph.number_of_edges(), 14)
        self.assertEqual(len(data.disruptions), 5)

    def test_missing_optional_duration_is_none(self):
        durations = {d["disruption_id"]: d["duration_days"] for d in sample().disruptions}
        self.assertIsNone(durations["D005"])
        self.assertEqual(durations["D001"], 14)

    def test_duplicate_entity_strict_raises_lenient_drops(self):
        ents = [["A", "A1", "supplier", "X"], ["A", "A2", "supplier", "X"], ["F", "F1", "factory", "X"]]
        rels = [["A", "F", "supplies", "0.5", "steel", "false"]]
        with self.assertRaises(GraphValidationError):
            build_supply_chain(*frames(ents, rels)[:2], strict=True)
        data = build_supply_chain(*frames(ents, rels)[:2], strict=False)
        self.assertEqual(data.graph.number_of_nodes(), 2)
        self.assertFalse(data.report.is_valid)

    def test_wrong_relationship_direction_is_an_error(self):
        ents = [["S", "S1", "supplier", "X"], ["W", "W1", "warehouse", "X"]]
        rels = [["S", "W", "supplies", "0.5", "steel", "false"]]   # supplies must end at a factory
        with self.assertRaises(GraphValidationError):
            build_supply_chain(*frames(ents, rels)[:2])

    def test_dependency_score_out_of_range(self):
        ents = [["S", "S1", "supplier", "X"], ["F", "F1", "factory", "X"]]
        rels = [["S", "F", "supplies", "1.5", "steel", "false"]]
        with self.assertRaises(GraphValidationError):
            build_supply_chain(*frames(ents, rels)[:2])

    def test_cycle_is_detected(self):
        # Two ships_to rows would fail the type rule first, so build the cycle through
        # the clean-record path used after validation.
        records = [{"source_id": "F1", "target_id": "F2", "relationship_type": "ships_to",
                    "dependency_score": 0.5, "material": None, "is_backup": False, "_row": 2},
                   {"source_id": "F2", "target_id": "F1", "relationship_type": "ships_to",
                    "dependency_score": 0.5, "material": None, "is_backup": False, "_row": 3}]
        report = graph_utils.ValidationReport()
        from graph_builder import _check_relationship_consistency
        with self.assertRaises(GraphValidationError):
            _check_relationship_consistency(records, report)
        self.assertFalse(report.is_valid)

    def test_backup_without_matching_primary_material_warns(self):
        ents = [["S1", "S1", "supplier", "X"], ["S2", "S2", "supplier", "X"], ["F", "F", "factory", "X"]]
        rels = [["S1", "F", "supplies", "0.5", "steel", "false"],
                ["S2", "F", "supplies", "0.5", "plastic", "true"]]
        data = build_supply_chain(*frames(ents, rels)[:2])
        self.assertEqual(len(data.report.warnings), 1)
        self.assertIn("no effect", data.report.warnings[0].message)

    def test_invalid_date_is_an_error(self):
        ents = [["S", "S", "supplier", "X"]]
        disr = [["D1", "S", "flood", "0.5", "X", "not-a-date", ""]]
        e, r, d = frames(ents, [], disr)
        r = pd.DataFrame(columns=["source_id", "target_id", "relationship_type", "dependency_score"])
        with self.assertRaises(GraphValidationError):
            build_supply_chain(e, r, d)


# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------

class WindowTests(unittest.TestCase):
    def test_window_is_inclusive(self):
        self.assertEqual(ia.disruption_window(disruption("D", "S", 0.5, "2026-11-10", 14)),
                         (date(2026, 11, 10), date(2026, 11, 23)))

    def test_open_ended_and_unknown_start(self):
        self.assertEqual(ia.disruption_window(disruption("D", "S", 0.5, "2026-11-22", None)),
                         (date(2026, 11, 22), None))
        self.assertEqual(ia.disruption_window(disruption("D", "S", 0.5, None, 5)), (None, None))

    def test_zero_duration_counts_as_one_day(self):
        self.assertEqual(ia.disruption_window(disruption("D", "S", 0.5, "2026-11-10", 0)),
                         (date(2026, 11, 10), date(2026, 11, 10)))

    def test_overlap_rules(self):
        a = (date(2026, 1, 1), date(2026, 1, 10))
        self.assertTrue(ia.windows_overlap(a, (date(2026, 1, 10), date(2026, 1, 20))))   # share one day
        self.assertFalse(ia.windows_overlap(a, (date(2026, 1, 11), date(2026, 1, 20))))  # adjacent, no overlap
        self.assertTrue(ia.windows_overlap(a, (date(2026, 1, 5), None)))                 # open-ended
        self.assertTrue(ia.windows_overlap(a, (None, None)))                             # unknown timing

    def test_invalid_date_text_raises(self):
        with self.assertRaises(ValueError):
            ia.disruption_window(disruption("D", "S", 0.5, "10/11/2026", 3))

    def test_active_disruptions_filter(self):
        ds = sample().disruptions
        ids = lambda start, end: [d["disruption_id"] for d in ia.active_disruptions(ds, start, end)]
        self.assertEqual(ids(None, None), ["D001", "D002", "D003", "D004", "D005"])
        self.assertEqual(ids("2026-11-26", "2026-12-25"), ["D004", "D005"])   # D001-D003 ended
        self.assertEqual(ids("2026-11-01", "2026-11-09"), [])                 # nothing started yet
        with self.assertRaises(ValueError):
            ia.active_disruptions(ds, "2026-12-01", "2026-11-01")


# ---------------------------------------------------------------------------
# Impact analysis
# ---------------------------------------------------------------------------

class ImpactTests(unittest.TestCase):
    def setUp(self):
        data = sample()
        self.graph, self.ds = data.graph, data.disruptions
        self.by_id = {d["disruption_id"]: d for d in self.ds}

    @staticmethod
    def scores(result):
        return {a["entity_id"]: a["impact_score"] for a in result["affected_entities"]}

    def test_isolated_analysis_keeps_backup_active(self):
        # S001 -> F001 loses 0.9*0.6 = 0.54, but backup S004 replaces 0.5 of it: 0.27
        result = ia.analyze_disruption(self.graph, self.by_id["D001"])
        self.assertEqual(self.scores(result), {"F001": 0.27, "W001": 0.162, "W002": 0.108})
        self.assertEqual(result["concurrent_disruption_ids"], [])

    def test_overlapping_disruption_weakens_the_backup(self):
        # D004 (S004, severity 0.8) overlaps D001, so the backup only delivers 20%:
        # mitigation 0.5*0.2 = 0.1 -> F001 = 0.9*0.6*0.9 = 0.486
        result = ia.analyze_disruption(self.graph, self.by_id["D001"], concurrent_disruptions=self.ds)
        self.assertAlmostEqual(self.scores(result)["F001"], 0.486, places=4)
        self.assertIn("D004", result["concurrent_disruption_ids"])
        self.assertNotIn("D001", result["concurrent_disruption_ids"])

    def test_non_overlapping_disruption_is_ignored(self):
        later = dict(self.by_id["D004"], start_date="2027-03-01")
        result = ia.analyze_disruption(self.graph, self.by_id["D001"],
                                       concurrent_disruptions=[self.by_id["D001"], later])
        self.assertEqual(self.scores(result)["F001"], 0.27)
        self.assertEqual(result["concurrent_disruption_ids"], [])

    def test_unknown_start_date_is_treated_as_overlapping(self):
        unknown = dict(self.by_id["D004"], start_date=None, duration_days=None)
        result = ia.analyze_disruption(self.graph, self.by_id["D001"], concurrent_disruptions=[unknown])
        self.assertAlmostEqual(self.scores(result)["F001"], 0.486, places=4)

    def test_fully_hit_backup_gives_no_mitigation(self):
        full = dict(self.by_id["D004"], severity=1.0)
        result = ia.analyze_disruption(self.graph, self.by_id["D001"], concurrent_disruptions=[full])
        self.assertAlmostEqual(self.scores(result)["F001"], 0.54, places=4)   # 0.9 * 0.6, no relief

    def test_disrupted_backup_itself_gets_no_credit(self):
        # D004 disrupts S004 directly; F001 is not downstream of S004 through normal flow,
        # so only F003 (normal supplier path) is affected.
        result = ia.analyze_disruption(self.graph, self.by_id["D004"])
        self.assertEqual(set(self.scores(result)), {"F003", "W001", "W003"})

    def test_entity_with_no_downstream(self):
        result = ia.analyze_disruption(self.graph, self.by_id["D005"])
        self.assertEqual(result["affected_entities"], [])
        self.assertEqual(result["summary"]["highest_risk_level"], "none")

    def test_result_fields(self):
        result = ia.analyze_disruption(self.graph, self.by_id["D001"])
        self.assertEqual(result["start_date"], "2026-11-10")
        self.assertEqual(result["end_date"], "2026-11-23")
        self.assertIsNone(ia.analyze_disruption(self.graph, self.by_id["D005"])["end_date"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ia.analyze_disruption(self.graph, disruption("X", "S001", 1.5))
        with self.assertRaises(EntityNotFoundError):
            ia.analyze_disruption(self.graph, disruption("X", "NOPE", 0.5))
        with self.assertRaises(ValueError):
            ia.analyze_disruption(self.graph, {"disruption_id": "X", "entity_id": "S001"})

    def test_scores_are_capped_at_one(self):
        g = build_supply_chain(*frames(
            [["S1", "S1", "supplier", "X"], ["S2", "S2", "supplier", "X"], ["F", "F", "factory", "X"]],
            [["S1", "F", "supplies", "0.9", "a", "false"], ["S2", "F", "supplies", "0.9", "b", "false"]]
        )[:2]).graph
        # Both suppliers fully disrupted at once is two disruptions; each alone gives 0.9.
        self.assertEqual(ia.calculate_impact_scores(g, "S1", 1.0)["F"], 0.9)

    def test_combined_impact_uses_product_formula(self):
        combined = {c["entity_id"]: c for c in ia.combine_impacts(self.graph, self.ds)}
        # F001: D001 -> 0.486 (backup weakened by D004), D002 -> 0.6*0.3 = 0.18
        self.assertAlmostEqual(combined["F001"]["combined_impact_score"],
                               1 - (1 - 0.486) * (1 - 0.18), places=4)
        self.assertEqual(combined["F001"]["disruption_ids"], ["D001", "D002"])
        self.assertTrue(combined["S001"]["directly_disrupted"])
        self.assertFalse(combined["F001"]["directly_disrupted"])

    def test_combined_impact_respects_window(self):
        ids = lambda c: {e["entity_id"] for e in c}
        late = ia.combine_impacts(self.graph, self.ds, "2026-11-26", "2026-12-25")
        self.assertNotIn("S001", ids(late))        # D001 ended 2026-11-23
        self.assertIn("S004", ids(late))
        self.assertIn("W003", ids(late))

    def test_window_filter_still_checks_overlap_against_all_disruptions(self):
        # Window selects only D001, but D004 still overlaps it in time and weakens the backup.
        analysis = ia.analyze_disruptions(self.graph, self.ds, window_start="2026-11-10",
                                          window_end="2026-11-12")
        self.assertEqual([r["disruption_id"] for r in analysis["results"]], ["D001"])
        self.assertAlmostEqual(self.scores(analysis["results"][0])["F001"], 0.486, places=4)
        self.assertEqual(analysis["time_window"], {"start": "2026-11-10", "end": "2026-11-12"})

    def test_analyze_disruptions_defaults(self):
        analysis = ia.analyze_disruptions(self.graph, self.ds, data_label="x")
        self.assertEqual(analysis["schema_version"], graph_utils.SCHEMA_VERSION)
        self.assertEqual(analysis["disruption_count"], 5)
        self.assertEqual(analysis["time_window"], {"start": None, "end": None})

    def test_timeline_horizons(self):
        timeline = ia.project_timeline(self.graph, self.ds, "2026-11-26")
        self.assertEqual([t["horizon_days"] for t in timeline], [30, 60, 90])
        self.assertEqual(timeline[0]["window_start"], "2026-11-26")
        self.assertEqual(timeline[0]["window_end"], "2026-12-25")      # 30 days inclusive
        for step in timeline:
            self.assertEqual(step["disruption_ids"], ["D004", "D005"])  # D001-D003 already ended

    def test_timeline_picks_up_later_starts_in_longer_horizons(self):
        future = self.ds + [disruption("D006", "S003", 0.7, "2027-01-15", 10)]
        t30, t60, t90 = ia.project_timeline(self.graph, future, "2026-11-26")
        self.assertNotIn("D006", t30["disruption_ids"])
        self.assertIn("D006", t60["disruption_ids"])
        self.assertIn("D006", t90["disruption_ids"])

    def test_timeline_validation(self):
        with self.assertRaises(ValueError):
            ia.project_timeline(self.graph, self.ds, None)
        with self.assertRaises(ValueError):
            ia.project_timeline(self.graph, self.ds, "2026-11-26", horizons=[0])


# ---------------------------------------------------------------------------
# ML tables
# ---------------------------------------------------------------------------

class FeatureTableTests(unittest.TestCase):
    def setUp(self):
        data = sample()
        self.graph = data.graph
        self.analysis = ia.analyze_disruptions(self.graph, data.disruptions)

    def test_pairs_table_only_has_affected_entities(self):
        df = ia.results_to_dataframe(self.analysis)
        self.assertEqual(len(df), sum(len(r["affected_entities"]) for r in self.analysis["results"]))
        self.assertNotIn("S003", set(df["affected_entity_id"]))

    def test_one_row_per_entity_including_unaffected(self):
        df = ia.node_features_dataframe(self.graph, self.analysis).set_index("entity_id")
        self.assertEqual(len(df), 12)
        for unaffected in ("S003", "S005", "S006"):
            self.assertEqual(df.loc[unaffected, "combined_impact_score"], 0.0)
            self.assertEqual(df.loc[unaffected, "risk_level"], "none")
            self.assertEqual(df.loc[unaffected, "n_disruptions"], 0)
            self.assertTrue(math.isnan(df.loc[unaffected, "min_hops"]))

    def test_structural_features(self):
        df = ia.node_features_dataframe(self.graph).set_index("entity_id")
        self.assertNotIn("combined_impact_score", df.columns)       # no analysis -> structure only
        self.assertEqual(df.loc["F001", "in_degree"], 2)             # backup edge not counted
        self.assertEqual(df.loc["F001", "out_degree"], 2)
        self.assertEqual(df.loc["F001", "backup_in_degree"], 1)
        self.assertEqual(df.loc["F001", "has_backup"], 1)
        self.assertAlmostEqual(df.loc["F001", "backup_coverage"], 0.6 / 0.9, places=3)  # steel covered
        self.assertAlmostEqual(df.loc["F001", "total_dependency_in"], 0.9)
        self.assertEqual(df.loc["F001", "upstream_count"], 2)        # S001, S002
        self.assertEqual(df.loc["S004", "backup_out_degree"], 1)
        self.assertEqual(df.loc["S004", "out_degree"], 1)            # only its normal edge to F003
        self.assertEqual(df.loc["F002", "has_backup"], 0)
        self.assertEqual(df.loc["F002", "backup_coverage"], 0.0)
        self.assertEqual(df.loc["S001", "in_degree"], 0)
        self.assertEqual(df.loc["S001", "backup_coverage"], 0.0)
        self.assertEqual(df.loc["W001", "downstream_count"], 0)

    def test_impact_features(self):
        df = ia.node_features_dataframe(self.graph, self.analysis).set_index("entity_id")
        self.assertEqual(df.loc["S001", "directly_disrupted"], 1)
        self.assertEqual(df.loc["S001", "min_hops"], 0)
        self.assertEqual(df.loc["S001", "max_single_impact"], 0.9)
        self.assertEqual(df.loc["F001", "directly_disrupted"], 0)
        self.assertEqual(df.loc["F001", "min_hops"], 1)
        self.assertAlmostEqual(df.loc["F001", "max_single_impact"], 0.486, places=4)
        self.assertEqual(df.loc["W003", "n_disruptions"], 4)

    def test_per_disruption_table_covers_every_entity(self):
        df = ia.node_features_dataframe(self.graph, self.analysis, per_disruption=True)
        self.assertEqual(len(df), 5 * 12)
        d3 = df[df["disruption_id"] == "D003"].set_index("entity_id")        # F002 power outage
        self.assertEqual(d3.loc["F002", "is_disrupted_entity"], 1)
        self.assertEqual(d3.loc["F002", "impact_score"], 0.5)
        self.assertEqual(d3.loc["F002", "hops"], 0)
        self.assertEqual(d3.loc["W002", "impact_score"], 0.3)
        self.assertEqual(d3.loc["S001", "impact_score"], 0.0)                # untouched -> zero
        self.assertEqual(d3.loc["S001", "risk_level"], "none")
        self.assertTrue(math.isnan(d3.loc["S001", "hops"]))
        self.assertEqual(d3.loc["F002", "n_concurrent"], 3)   # D001, D002, D004 (D005 starts later)

    def test_per_disruption_needs_analysis(self):
        with self.assertRaises(ValueError):
            ia.node_features_dataframe(self.graph, None, per_disruption=True)


# ---------------------------------------------------------------------------
# Cypher export and live events (fake driver, no database)
# ---------------------------------------------------------------------------

class FakeTx:
    def __init__(self, log):
        self.log = log

    def run(self, query, params=None, **kwargs):
        self.log.append(("tx", query, params))


class FakeSession:
    def __init__(self, log, database):
        self.log, self.database = log, database

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, params=None, **kwargs):
        self.log.append(("auto", query, params))

    def execute_write(self, fn):
        self.log.append(("begin", None, None))
        fn(FakeTx(self.log))
        self.log.append(("commit", None, None))


class FakeDriver:
    def __init__(self):
        self.log, self.database = [], None

    def session(self, database=None):
        self.database = database
        return FakeSession(self.log, database)


class CypherTests(unittest.TestCase):
    def setUp(self):
        data = sample()
        self.graph, self.ds = data.graph, data.disruptions
        self.by_id = {d["disruption_id"]: d for d in self.ds}

    def test_graph_statements(self):
        statements = ns.graph_to_cypher(self.graph)
        queries = [q for q, _ in statements]
        self.assertTrue(queries[0].startswith("CREATE CONSTRAINT"))
        self.assertTrue(queries[1].startswith("CREATE CONSTRAINT"))
        node_stmts = [(q, p) for q, p in statements if "MERGE (e:Entity" in q]
        self.assertEqual(len(node_stmts), 3)
        self.assertEqual(sum(len(p["rows"]) for _, p in node_stmts), 12)
        rel_stmts = {q.split("[r:")[1].split("]")[0]: p for q, p in statements if "MERGE (a)-[r:" in q}
        self.assertEqual(set(rel_stmts), {"SUPPLIES", "SHIPS_TO"})
        self.assertEqual(len(rel_stmts["SUPPLIES"]["rows"]), 8)
        self.assertEqual(len(rel_stmts["SHIPS_TO"]["rows"]), 6)
        backups = [r for r in rel_stmts["SUPPLIES"]["rows"] if r["props"]["is_backup"]]
        self.assertEqual([(r["source_id"], r["target_id"]) for r in backups], [("S004", "F001")])

    def test_resync_does_not_reset_live_risk_state(self):
        for query, _ in ns.graph_to_cypher(self.graph):
            if "MERGE (e:Entity" in query:
                before_on_create, _, after = query.partition("ON CREATE SET")
                self.assertIn("risk_score", after.split("SET e:")[0])   # only inside ON CREATE
                self.assertNotIn("risk_score", after.split("SET e:")[1])

    def test_no_values_in_query_text(self):
        for query, _ in ns.graph_to_cypher(self.graph):
            for value in ("S001", "Chennai", "Eastern Steel Works", "steel"):
                self.assertNotIn(value, query)

    def test_unsafe_identifiers_are_rejected(self):
        with self.assertRaises(ValueError):
            ns._safe_identifier("Supplier) DETACH DELETE (n")
        g = copy.deepcopy(self.graph)
        g.nodes["S001"]["entity_type"] = "x}) DETACH DELETE (n"
        with self.assertRaises(ValueError):
            ns.graph_to_cypher(g)

    def test_risk_state_covers_every_entity(self):
        combined = ia.combine_impacts(self.graph, self.ds)
        (query, params), = ns.risk_state_statements(self.graph, combined, updated_at="T")
        rows = {r["entity_id"]: r for r in params["rows"]}
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows["S003"]["risk_score"], 0.0)               # unaffected -> reset
        self.assertEqual(rows["S003"]["risk_level"], "none")
        self.assertEqual(rows["S003"]["disruption_ids"], [])
        self.assertEqual(rows["S001"]["risk_level"], "high")
        self.assertTrue(rows["S001"]["directly_disrupted"])
        self.assertEqual(rows["S001"]["updated_at"], "T")

    def test_apply_event_dry_run(self):
        result = ns.apply_event(self.graph, self.by_id["D004"], updated_at="T")
        self.assertEqual(result["event_result"]["disruption_id"], "D004")
        self.assertEqual([e["disruption_id"] for e in result["active_events"]], ["D004"])
        queries = [q for q, _ in result["statements"]]
        self.assertTrue(queries[0].startswith("MERGE (d:Disruption"))
        self.assertIn("DELETE r", queries[1])
        self.assertTrue(queries[-1].startswith("UNWIND $rows AS row\nMATCH (e:Entity"))
        impacts = result["statements"][3][1]["rows"]
        self.assertEqual({r["entity_id"] for r in impacts}, {"F003", "W001", "W003"})

    def test_apply_event_combines_with_active_events(self):
        first = ns.apply_event(self.graph, self.by_id["D001"])
        second = ns.apply_event(self.graph, self.by_id["D004"], active_events=first["active_events"])
        self.assertEqual([e["disruption_id"] for e in second["active_events"]], ["D001", "D004"])
        combined = {c["entity_id"]: c for c in second["combined_impact"]}
        # D004 overlaps D001, so F001 is hit at 0.486 (not 0.27) in the combined view
        self.assertAlmostEqual(combined["F001"]["combined_impact_score"], 0.486, places=4)
        self.assertEqual(second["event_result"]["concurrent_disruption_ids"], ["D001"])

    def test_apply_event_replaces_same_id(self):
        first = ns.apply_event(self.graph, self.by_id["D001"])
        weaker = dict(self.by_id["D001"], severity=0.3)
        second = ns.apply_event(self.graph, weaker, active_events=first["active_events"])
        self.assertEqual(len(second["active_events"]), 1)
        self.assertEqual(second["active_events"][0]["severity"], 0.3)

    def test_apply_event_as_of_hides_ended_events(self):
        # On 2026-12-20: D001 ended 11-23 and D004 ended 12-10; D005 is open-ended.
        events = [self.by_id["D001"], self.by_id["D004"]]
        result = ns.apply_event(self.graph, self.by_id["D005"], active_events=events, as_of="2026-12-20")
        ids = {c["entity_id"] for c in result["combined_impact"]}
        self.assertEqual(ids, {"W003"})
        # The event is still returned as active input for the caller to carry forward.
        self.assertEqual(len(result["active_events"]), 3)

    def test_apply_event_validation_happens_before_any_write(self):
        driver = FakeDriver()
        with self.assertRaises(EntityNotFoundError):
            ns.apply_event(self.graph, disruption("D9", "NOPE", 0.5), driver=driver)
        with self.assertRaises(ValueError):
            ns.apply_event(self.graph, disruption("D9", "S001", 7), driver=driver)
        with self.assertRaises(ValueError):
            ns.apply_event(self.graph, {"entity_id": "S001", "severity": 0.5}, driver=driver)
        self.assertEqual(driver.log, [])

    def test_run_statements_splits_schema_from_data(self):
        driver = FakeDriver()
        ns.sync_graph(self.graph, driver, database="supply")
        self.assertEqual(driver.database, "supply")
        kinds = [k for k, _, _ in driver.log]
        self.assertEqual(kinds[:2], ["auto", "auto"])                    # constraints outside the transaction
        self.assertEqual(kinds[2], "begin")
        self.assertEqual(kinds[-1], "commit")
        self.assertEqual(kinds.count("begin"), 1)                        # all data writes in ONE transaction
        self.assertEqual(kinds.count("tx"), len(ns.graph_to_cypher(self.graph)) - 2)

    def test_apply_event_with_driver_runs_in_one_transaction(self):
        driver = FakeDriver()
        ns.apply_event(self.graph, self.by_id["D004"], driver=driver, updated_at="T")
        kinds = [k for k, _, _ in driver.log]
        self.assertEqual(kinds[0], "begin")
        self.assertEqual(kinds[-1], "commit")
        self.assertEqual(kinds.count("tx"), 5)

    def test_sync_risk_state_clears_ended_events(self):
        result = ns.sync_risk_state(self.graph, [], updated_at="T")      # nothing active
        rows = result["statements"][0][1]["rows"]
        self.assertTrue(all(r["risk_score"] == 0.0 and r["risk_level"] == "none" for r in rows))

    def test_export_statements_roundtrip(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = ns.export_statements(ns.graph_to_cypher(self.graph), Path(tmp) / "out" / "c.json")
            loaded = json.loads(path.read_text())
        self.assertEqual(len(loaded), len(ns.graph_to_cypher(self.graph)))
        self.assertEqual(set(loaded[0]), {"query", "parameters"})


# ---------------------------------------------------------------------------
# Supplier -> SKU -> Warehouse graph (built from purchase-order records)
# ---------------------------------------------------------------------------
# Sample (synthetic) graph:
#   SUP-DEMO-01 --0.6--> SKU-DEMO-A --0.4--> WH-DEMO-N        A has TWO suppliers (01, 02)
#   SUP-DEMO-02 --0.4--> SKU-DEMO-A --0.6--> WH-DEMO-S
#   SUP-DEMO-02 --1.0--> SKU-DEMO-B --0.6--> WH-DEMO-N        B has a BACKUP supplier (03, 0.5)
#   SUP-DEMO-04 --1.0--> SKU-DEMO-C --0.4--> WH-DEMO-S

SUP1, SUP2, SUP3, SUP4 = "SUP-DEMO-01", "SUP-DEMO-02", "SUP-DEMO-03", "SUP-DEMO-04"
SKU_A, SKU_B, SKU_C = "SKU-DEMO-A", "SKU-DEMO-B", "SKU-DEMO-C"
WH_N, WH_S = "WH-DEMO-N", "WH-DEMO-S"


class PurchaseOrderGraphTests(unittest.TestCase):
    def setUp(self):
        data = po_sample()
        self.graph, self.ds = data.graph, data.disruptions
        self.by_id = {d["disruption_id"]: d for d in self.ds}

    def edge(self, u, v):
        return self.graph.edges[u, v]

    # -- construction ----------------------------------------------------------------
    def test_sample_loads_cleanly(self):
        data = po_sample()
        self.assertTrue(data.report.is_valid)
        self.assertEqual(len(data.report.issues), 0)
        self.assertEqual(graph_summary(data.graph), {
            "total_entities": 9, "suppliers": 4, "factories": 0, "skus": 3, "warehouses": 2,
            "total_relationships": 9, "backup_relationships": 1})

    def test_ids_are_taken_verbatim_from_the_purchase_orders(self):
        po = pd.read_csv(PO_DIR / "purchase_orders.csv", dtype=str)
        expected = set(po["supplier_id"]) | set(po["sku_id"]) | set(po["warehouse_id"])
        self.assertEqual(set(self.graph.nodes), expected)        # nothing renamed or invented
        self.assertFalse(any(n.startswith(("S0", "F0", "W0")) for n in self.graph.nodes))
        self.assertEqual(graph_utils.get_entity_ids_by_type(self.graph, "factory"), [])  # no factory link made up

    def test_edges_follow_the_schema(self):
        for u, v, d in self.graph.edges(data=True):
            pair = (self.graph.nodes[u]["entity_type"], self.graph.nodes[v]["entity_type"])
            self.assertIn(pair, graph_utils.RELATIONSHIP_RULES[d["relationship_type"]])
            self.assertIn(pair, [("supplier", "sku"), ("sku", "warehouse")])

    def test_shares_come_from_order_quantities(self):
        self.assertAlmostEqual(self.edge(SUP1, SKU_A)["dependency_score"], 0.6)    # 600 of 1000
        self.assertAlmostEqual(self.edge(SUP2, SKU_A)["dependency_score"], 0.4)    # 400 of 1000
        self.assertAlmostEqual(self.edge(SKU_A, WH_N)["dependency_score"], 0.4)    # 400 of 1000 into WH-N
        self.assertAlmostEqual(self.edge(SKU_B, WH_N)["dependency_score"], 0.6)    # 600 of 1000 (backup PO counts)
        for sku in (SKU_A, SKU_B, SKU_C):                                         # normal shares add up to 1
            total = sum(d["dependency_score"] for _, _, d in self.graph.in_edges(sku, data=True)
                        if not d["is_backup"])
            self.assertAlmostEqual(total, 1.0)
        for wh in (WH_N, WH_S):
            self.assertAlmostEqual(sum(d["dependency_score"] for _, _, d in self.graph.in_edges(wh, data=True)), 1.0)

    def test_backup_supplier_edge(self):
        e = self.edge(SUP3, SKU_B)
        self.assertTrue(e["is_backup"])
        self.assertAlmostEqual(e["dependency_score"], 0.5)        # 200 backup vs 400 normal
        self.assertEqual(e["material"], SKU_B)                    # a SKU is its own material

    def test_names_and_locations_come_from_the_orders(self):
        self.assertEqual(self.graph.nodes[SUP1]["location"], "Chennai")
        self.assertEqual(self.graph.nodes[WH_N]["entity_name"], "Demo North Warehouse")
        self.assertEqual(self.graph.nodes[SKU_A]["entity_type"], "sku")

    # -- retrieval helpers -----------------------------------------------------------
    def test_suppliers_of_sku_multiple_suppliers(self):
        self.assertEqual(graph_utils.get_suppliers_of_sku(self.graph, SKU_A), [SUP1, SUP2])
        self.assertEqual(graph_utils.get_suppliers_of_sku(self.graph, SKU_C), [SUP4])

    def test_backup_suppliers_are_separate(self):
        self.assertEqual(graph_utils.get_suppliers_of_sku(self.graph, SKU_B), [SUP2])
        self.assertEqual(graph_utils.get_suppliers_of_sku(self.graph, SKU_B, include_backup=True), [SUP2, SUP3])
        self.assertEqual(graph_utils.get_backup_suppliers_of_sku(self.graph, SKU_B), [SUP3])
        self.assertEqual(graph_utils.get_backup_suppliers_of_sku(self.graph, SKU_A), [])

    def test_warehouses_of_sku_and_reverse(self):
        self.assertEqual(graph_utils.get_warehouses_of_sku(self.graph, SKU_A), [WH_N, WH_S])
        self.assertEqual(graph_utils.get_warehouses_of_sku(self.graph, SKU_B), [WH_N])
        self.assertEqual(graph_utils.get_skus_at_warehouse(self.graph, WH_N), [SKU_A, SKU_B])
        self.assertEqual(graph_utils.get_skus_at_warehouse(self.graph, WH_S), [SKU_A, SKU_C])

    def test_skus_supplied_by(self):
        self.assertEqual(graph_utils.get_skus_supplied_by(self.graph, SUP2), [SKU_A, SKU_B])
        self.assertEqual(graph_utils.get_skus_supplied_by(self.graph, SUP3), [])               # only a backup
        self.assertEqual(graph_utils.get_skus_supplied_by(self.graph, SUP3, include_backup=True), [SKU_B])

    def test_helpers_reject_unknown_entities(self):
        for fn in (graph_utils.get_suppliers_of_sku, graph_utils.get_backup_suppliers_of_sku,
                   graph_utils.get_warehouses_of_sku):
            with self.assertRaises(EntityNotFoundError):
                fn(self.graph, "SKU-NOPE")

    def test_downstream_traversal_over_multiple_hops(self):
        g = self.graph
        self.assertEqual(graph_utils.get_downstream_entities(g, SUP1), [SKU_A, WH_N, WH_S])
        self.assertEqual(graph_utils.get_downstream_entities(g, SUP1, "warehouse"), [WH_N, WH_S])
        self.assertEqual(graph_utils.get_downstream_entities(g, SUP2), [SKU_A, SKU_B, WH_N, WH_S])
        self.assertEqual(graph_utils.get_downstream_entities(g, SKU_C), [WH_S])
        self.assertEqual(graph_utils.get_downstream_entities(g, WH_N), [])
        # backup edges are not normal flow: SUP3 reaches nothing unless backups are included
        self.assertEqual(graph_utils.get_downstream_entities(g, SUP3), [])
        self.assertEqual(graph_utils.get_downstream_entities(g, SUP3, include_backup=True), [SKU_B, WH_N])

    def test_upstream_traversal_over_multiple_hops(self):
        g = self.graph
        self.assertEqual(graph_utils.get_upstream_entities(g, WH_N), [SKU_A, SKU_B, SUP1, SUP2])
        self.assertEqual(graph_utils.get_upstream_entities(g, WH_N, "supplier"), [SUP1, SUP2])
        self.assertEqual(graph_utils.get_upstream_entities(g, WH_N, "supplier", include_backup=True),
                         [SUP1, SUP2, SUP3])
        self.assertEqual(graph_utils.get_upstream_entities(g, WH_S, "supplier"), [SUP1, SUP2, SUP4])

    # -- impact propagation ----------------------------------------------------------
    @staticmethod
    def scores(result):
        return {a["entity_id"]: a["impact_score"] for a in result["affected_entities"]}

    def test_supplier_impact_reaches_warehouses_in_two_hops(self):
        result = ia.analyze_disruption(self.graph, self.by_id["D-DEMO-1"])      # SUP1, severity 0.8
        self.assertEqual(self.scores(result), {SKU_A: 0.48, WH_N: 0.192, WH_S: 0.288})
        hops = {a["entity_id"]: a["hops"] for a in result["affected_entities"]}
        self.assertEqual(hops, {SKU_A: 1, WH_N: 2, WH_S: 2})
        self.assertEqual(result["propagation_paths"],
                         [[SUP1, SKU_A], [SUP1, SKU_A, WH_N], [SUP1, SKU_A, WH_S]])
        self.assertEqual(result["summary"]["affected_by_type"], {"sku": 1, "warehouse": 2})

    def test_second_supplier_of_a_sku_is_not_hit(self):
        result = ia.analyze_disruption(self.graph, self.by_id["D-DEMO-1"])
        self.assertNotIn(SUP2, self.scores(result))
        self.assertNotIn(SKU_B, self.scores(result))              # SUP1 does not supply B

    def test_backup_supplier_mitigates_a_sku(self):
        # SUP2 (severity 1.0) is B's only normal supplier; backup SUP3 replaces 0.5 of it.
        result = ia.analyze_disruption(self.graph, self.by_id["D-DEMO-2"])      # isolated
        self.assertEqual(self.scores(result),
                         {SKU_B: 0.5, SKU_A: 0.4, WH_N: 0.46, WH_S: 0.24})
        # WH-N is hit through two SKUs: 0.4*0.4 (via A) + 0.5*0.6 (via B)
        self.assertIn([SUP2, SKU_A, WH_N], result["propagation_paths"])
        self.assertIn([SUP2, SKU_B, WH_N], result["propagation_paths"])

    def test_overlapping_disruption_of_the_backup_removes_mitigation(self):
        result = ia.analyze_disruption(self.graph, self.by_id["D-DEMO-2"], concurrent_disruptions=self.ds)
        self.assertIn("D-DEMO-3", result["concurrent_disruption_ids"])
        self.assertEqual(self.scores(result)[SKU_B], 1.0)         # backup SUP3 is out as well
        self.assertAlmostEqual(self.scores(result)[WH_N], 0.76)   # 0.4*0.4 + 1.0*0.6

    def test_disrupted_backup_supplier_affects_nothing(self):
        result = ia.analyze_disruption(self.graph, self.by_id["D-DEMO-3"])
        self.assertEqual(result["affected_entities"], [])

    def test_sku_disruption_goes_straight_to_warehouses(self):
        result = ia.analyze_disruption(self.graph, self.by_id["D-DEMO-4"])      # SKU C, 0.6
        self.assertEqual(self.scores(result), {WH_S: 0.24})
        self.assertEqual(result["propagation_paths"], [[SKU_C, WH_S]])

    def test_scores_stay_within_zero_and_one(self):
        analysis = ia.analyze_disruptions(self.graph, self.ds)
        for result in analysis["results"]:
            for a in result["affected_entities"]:
                self.assertTrue(0.0 < a["impact_score"] <= 1.0)
        for c in analysis["combined_impact"]:
            self.assertTrue(0.0 <= c["combined_impact_score"] <= 1.0)

    # -- structural features (unchanged logic, new structure) -------------------------
    def test_structural_features_on_the_sku_graph(self):
        df = ia.node_features_dataframe(self.graph).set_index("entity_id")
        self.assertEqual(len(df), 9)
        self.assertEqual(df.loc[SKU_A, "in_degree"], 2)
        self.assertEqual(df.loc[SKU_A, "out_degree"], 2)
        self.assertEqual(df.loc[SKU_A, "has_backup"], 0)
        self.assertEqual(df.loc[SKU_B, "backup_in_degree"], 1)
        self.assertEqual(df.loc[SKU_B, "has_backup"], 1)
        self.assertAlmostEqual(df.loc[SKU_B, "backup_coverage"], 1.0)       # B's only supplier is covered
        self.assertEqual(df.loc[SUP3, "backup_out_degree"], 1)
        self.assertEqual(df.loc[SUP3, "out_degree"], 0)
        self.assertEqual(df.loc[SUP2, "downstream_count"], 4)               # A, B, WH-N, WH-S
        self.assertEqual(df.loc[WH_N, "upstream_count"], 4)                 # A, B, SUP1, SUP2
        self.assertAlmostEqual(df.loc[WH_N, "total_dependency_in"], 1.0)

    def test_node_feature_table_with_analysis(self):
        analysis = ia.analyze_disruptions(self.graph, self.ds)
        df = ia.node_features_dataframe(self.graph, analysis).set_index("entity_id")
        self.assertEqual(df.loc[SUP4, "combined_impact_score"], 0.0)        # untouched supplier
        self.assertEqual(df.loc[WH_N, "min_hops"], 2)
        self.assertEqual(df.loc[SKU_A, "min_hops"], 1)
        per = ia.node_features_dataframe(self.graph, analysis, per_disruption=True)
        self.assertEqual(len(per), 4 * 9)

    # -- validation ------------------------------------------------------------------
    def test_hand_written_sku_tables_are_accepted(self):
        ents = [["S", "S", "supplier", "X"], ["K", "K", "sku", ""], ["W", "W", "warehouse", "X"]]
        rels = [["S", "K", "supplies", "1", "", "false"], ["K", "W", "ships_to", "1", "", "false"]]
        data = build_supply_chain(*frames(ents, rels)[:2])
        self.assertTrue(data.report.is_valid)
        self.assertEqual(len(data.report.warnings), 0)         # no 'missing location' for a SKU

    def test_wrong_sku_relationship_directions_are_errors(self):
        ents = [["S", "S", "supplier", "X"], ["K", "K", "sku", ""], ["W", "W", "warehouse", "X"]]
        for rels in ([["W", "K", "ships_to", "1", "", "false"]],       # warehouse -> sku
                     [["S", "W", "ships_to", "1", "", "false"]],       # supplier -> warehouse
                     [["K", "S", "supplies", "1", "", "false"]],       # sku -> supplier
                     [["K", "W", "supplies", "1", "", "false"]]):      # sku -> warehouse as 'supplies'
            with self.assertRaises(GraphValidationError, msg=rels):
                build_supply_chain(*frames(ents, rels)[:2])

    def test_error_message_lists_the_allowed_pairs(self):
        ents = [["S", "S", "supplier", "X"], ["W", "W", "warehouse", "X"]]
        with self.assertRaises(GraphValidationError) as ctx:
            build_supply_chain(*frames(ents, [["S", "W", "supplies", "1", "", "false"]])[:2])
        self.assertIn("supplier -> sku", str(ctx.exception))
        self.assertIn("supplier -> factory", str(ctx.exception))

    def test_unknown_entity_and_duplicate_id_still_rejected(self):
        ents = [["S", "S", "supplier", "X"], ["K", "K", "sku", ""]]
        with self.assertRaises(GraphValidationError):
            build_supply_chain(*frames(ents, [["S", "NOPE", "supplies", "1", "", "false"]])[:2])
        dup = ents + [["K", "again", "sku", ""]]
        with self.assertRaises(GraphValidationError):
            build_supply_chain(*frames(dup, [])[:2])

    def test_sku_dependency_score_range_and_duplicates_still_checked(self):
        ents = [["S", "S", "supplier", "X"], ["K", "K", "sku", ""]]
        with self.assertRaises(GraphValidationError):
            build_supply_chain(*frames(ents, [["S", "K", "supplies", "1.2", "", "false"]])[:2])
        twice = [["S", "K", "supplies", "0.5", "", "false"], ["S", "K", "supplies", "0.5", "", "false"]]
        with self.assertRaises(GraphValidationError):
            build_supply_chain(*frames(ents, twice)[:2])

    def test_over_allocated_sku_still_warns(self):
        ents = [["S1", "S1", "supplier", "X"], ["S2", "S2", "supplier", "X"], ["K", "K", "sku", ""]]
        rels = [["S1", "K", "supplies", "0.7", "", "false"], ["S2", "K", "supplies", "0.7", "", "false"]]
        data = build_supply_chain(*frames(ents, rels)[:2])
        self.assertEqual(len(data.report.warnings), 1)
        self.assertIn("add up to 1.40", data.report.warnings[0].message)

    def test_cycle_check_covers_sku_graphs(self):
        # supplier/sku/warehouse rules make a pure cycle impossible; a mixed graph that tries
        # one is still caught by the whole-table check.
        from graph_builder import _check_relationship_consistency
        rec = lambda u, v, row: {"source_id": u, "target_id": v, "relationship_type": "ships_to",
                                 "dependency_score": 0.5, "material": None, "is_backup": False, "_row": row}
        with self.assertRaises(GraphValidationError):
            _check_relationship_consistency([rec("K1", "K2", 2), rec("K2", "K1", 3)],
                                            graph_utils.ValidationReport())

    def test_backup_with_sku_default_material_has_no_warning(self):
        ents = [["S1", "S1", "supplier", "X"], ["S2", "S2", "supplier", "X"], ["K", "K", "sku", ""]]
        rels = [["S1", "K", "supplies", "1", "", "false"], ["S2", "K", "supplies", "0.5", "", "true"]]
        data = build_supply_chain(*frames(ents, rels)[:2])
        self.assertEqual(len(data.report.issues), 0)
        self.assertAlmostEqual(ia.calculate_impact_scores(data.graph, "S1", 1.0)["K"], 0.5)

    def test_legacy_factory_graph_still_works_next_to_skus(self):
        ents = [["S", "S", "supplier", "X"], ["F", "F", "factory", "X"], ["K", "K", "sku", ""],
                ["W", "W", "warehouse", "X"]]
        rels = [["S", "F", "supplies", "0.5", "steel", "false"], ["F", "W", "ships_to", "1", "", "false"],
                ["S", "K", "supplies", "1", "", "false"], ["K", "W", "ships_to", "0.5", "", "false"]]
        g = build_supply_chain(*frames(ents, rels)[:2]).graph
        self.assertEqual(graph_utils.get_downstream_entities(g, "S"), ["F", "K", "W"])
        self.assertEqual(graph_utils.get_suppliers_of_factory(g, "F"), ["S"])
        self.assertEqual(graph_utils.get_suppliers_of_sku(g, "K"), ["S"])
        self.assertAlmostEqual(ia.calculate_impact_scores(g, "S", 1.0)["W"], 0.5 * 1 + 1 * 0.5)

    def test_original_sample_is_unchanged(self):
        data = sample()
        self.assertEqual(graph_summary(data.graph)["skus"], 0)
        self.assertEqual(graph_utils.get_suppliers_of_factory(data.graph, "F001"), ["S001", "S002"])


class PurchaseOrderBuilderTests(unittest.TestCase):
    def test_shares_from_quantities(self):
        po = po_frame([("A", "K", "W1", "30"), ("B", "K", "W1", "10"), ("B", "K", "W2", "60")])
        g = build_supply_chain_from_purchase_orders(po).graph
        self.assertAlmostEqual(g.edges["A", "K"]["dependency_score"], 0.3)
        self.assertAlmostEqual(g.edges["B", "K"]["dependency_score"], 0.7)
        self.assertAlmostEqual(g.edges["K", "W1"]["dependency_score"], 1.0)
        self.assertAlmostEqual(g.edges["K", "W2"]["dependency_score"], 1.0)

    def test_repeated_orders_are_summed(self):
        po = po_frame([("A", "K", "W", "10"), ("A", "K", "W", "10"), ("B", "K", "W", "20")])
        g = build_supply_chain_from_purchase_orders(po).graph
        self.assertEqual(g.number_of_edges(), 3)
        self.assertAlmostEqual(g.edges["A", "K"]["dependency_score"], 0.5)

    def test_no_quantity_column_counts_order_lines(self):
        po = po_frame([("A", "K", "W", "1"), ("A", "K", "W", "1"), ("B", "K", "W", "1")]).drop(columns="quantity")
        data = build_supply_chain_from_purchase_orders(po)
        self.assertAlmostEqual(data.graph.edges["A", "K"]["dependency_score"], 2 / 3)
        self.assertTrue(any("no quantity column" in w.message for w in data.report.warnings))

    def test_column_mapping(self):
        po = pd.DataFrame({"Vendor": ["A", "B"], "Item": ["K", "K"], "DC": ["W", "W"], "Qty": ["1", "3"]})
        cols = {"supplier_id": "vendor", "sku_id": "item", "warehouse_id": "dc", "quantity": "qty"}
        g = build_supply_chain_from_purchase_orders(po, columns=cols).graph
        self.assertAlmostEqual(g.edges["B", "K"]["dependency_score"], 0.75)

    def test_missing_required_column_and_bad_mapping(self):
        with self.assertRaises(GraphValidationError) as ctx:
            build_supply_chain_from_purchase_orders(pd.DataFrame({"supplier_id": ["A"], "sku_id": ["K"]}))
        self.assertIn("warehouse_id", str(ctx.exception))
        with self.assertRaises(GraphValidationError):
            build_supply_chain_from_purchase_orders(po_frame([("A", "K", "W", "1")]), columns={"nope": "x"})

    def test_bad_rows_strict_raises_lenient_drops(self):
        po = po_frame([("A", "K", "W", "10"), ("", "K", "W", "5"), ("B", "K", "W", "-3"),
                       ("C", "K", "W", "abc"), ("D", "K", "W", "10", "maybe")])
        with self.assertRaises(GraphValidationError):
            build_supply_chain_from_purchase_orders(po)
        data = build_supply_chain_from_purchase_orders(po, strict=False)
        self.assertEqual(sorted(data.graph.nodes), ["A", "K", "W"])
        self.assertEqual(len(data.report.errors), 4)
        self.assertEqual({e.table for e in data.report.errors}, {"purchase_orders"})

    def test_id_shared_by_two_types_is_a_duplicate_error(self):
        po = po_frame([("X", "X", "W", "1")])             # same ID used as supplier and SKU
        with self.assertRaises(GraphValidationError):
            build_supply_chain_from_purchase_orders(po)

    def test_empty_orders_raise(self):
        with self.assertRaises(GraphValidationError):
            build_supply_chain_from_purchase_orders(po_frame([]))

    def test_backup_from_column_and_from_pairs(self):
        po = po_frame([("A", "K", "W", "100"), ("B", "K", "W", "40", "true")])
        by_column = build_supply_chain_from_purchase_orders(po).graph
        by_pairs = build_supply_chain_from_purchase_orders(po_frame([("A", "K", "W", "100"), ("B", "K", "W", "40")]),
                                                           backup_pairs=[("B", "K")]).graph
        for g in (by_column, by_pairs):
            self.assertTrue(g.edges["B", "K"]["is_backup"])
            self.assertAlmostEqual(g.edges["B", "K"]["dependency_score"], 0.4)
            self.assertAlmostEqual(g.edges["A", "K"]["dependency_score"], 1.0)   # backup volume not in the share
        # without any marking nobody is a backup
        plain = build_supply_chain_from_purchase_orders(po_frame([("A", "K", "W", "100"), ("B", "K", "W", "40")])).graph
        self.assertEqual(graph_summary(plain)["backup_relationships"], 0)

    def test_backup_with_no_normal_supplier_is_treated_as_normal(self):
        data = build_supply_chain_from_purchase_orders(po_frame([("B", "K", "W", "40", "true")]))
        self.assertFalse(data.graph.edges["B", "K"]["is_backup"])
        self.assertAlmostEqual(data.graph.edges["B", "K"]["dependency_score"], 1.0)
        self.assertTrue(any("no normal supplier" in w.message for w in data.report.warnings))

    def test_mixed_normal_and_backup_orders_count_as_normal(self):
        po = po_frame([("A", "K", "W", "60"), ("A", "K", "W", "40", "true")])
        data = build_supply_chain_from_purchase_orders(po)
        self.assertFalse(data.graph.edges["A", "K"]["is_backup"])
        self.assertTrue(any("both normal and backup" in w.message for w in data.report.warnings))

    def test_conflicting_names_warn_and_keep_first(self):
        po = po_frame([("A", "K", "W", "1"), ("A", "K", "W", "1")])
        po["supplier_name"] = ["Alpha", "Alpha Ltd"]
        data = build_supply_chain_from_purchase_orders(po)
        self.assertEqual(data.graph.nodes["A"]["entity_name"], "Alpha")
        self.assertTrue(any("more than one entity_name" in w.message for w in data.report.warnings))

    def test_disruptions_are_validated_against_po_entities(self):
        po = po_frame([("A", "K", "W", "1")])
        bad = pd.DataFrame([["D1", "S001", "flood", "0.5", "X", "2026-11-10", "3"]],
                           columns=["disruption_id", "entity_id", "disruption_type", "severity",
                                    "location", "start_date", "duration_days"])
        with self.assertRaises(GraphValidationError):           # legacy IDs are NOT silently mapped
            build_supply_chain_from_purchase_orders(po, bad)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_supply_chain_from_purchase_orders(PO_DIR / "missing.csv")

    def test_end_to_end_json_and_cypher(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "po_result.json"
            analysis = ia.run_analysis_from_purchase_orders(PO_DIR / "purchase_orders.csv",
                                                            PO_DIR / "disruptions.csv", out,
                                                            data_label="t", as_of="2026-11-16")
            saved = json.loads(out.read_text())
        self.assertEqual(saved["disruption_count"], 4)
        self.assertEqual(analysis["graph_summary"]["skus"], 3)
        self.assertEqual(analysis["graph_summary"]["backup_relationships"], 1)
        self.assertEqual(len(saved["timeline"]), 3)

        graph = po_sample().graph
        statements = ns.graph_to_cypher(graph)
        labels = [q.split("SET e:")[1].split("\n")[0] for q, _ in statements if "MERGE (e:Entity" in q]
        self.assertEqual(sorted(labels), ["Sku", "Supplier", "Warehouse"])
        rels = {q.split("[r:")[1].split("]")[0]: p for q, p in statements if "MERGE (a)-[r:" in q}
        self.assertEqual(len(rels["SUPPLIES"]["rows"]), 5)
        self.assertEqual(len(rels["SHIPS_TO"]["rows"]), 4)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class EndToEndTests(unittest.TestCase):
    def test_run_analysis_writes_json_and_timeline(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            analysis = ia.run_analysis(DATA_DIR / "entities.csv", DATA_DIR / "relationships.csv",
                                       DATA_DIR / "disruptions.csv", out, data_label="t",
                                       as_of="2026-11-26")
            saved = json.loads(out.read_text())
        self.assertEqual(saved["disruption_count"], 5)
        self.assertEqual(len(saved["timeline"]), 3)
        self.assertTrue(saved["validation"]["is_valid"])
        self.assertEqual(saved["graph_summary"]["backup_relationships"], 1)
        self.assertEqual(analysis["schema_version"], "1.1")

    def test_no_timeline_without_as_of(self):
        analysis = ia.run_analysis(DATA_DIR / "entities.csv", DATA_DIR / "relationships.csv",
                                   DATA_DIR / "disruptions.csv")
        self.assertNotIn("timeline", analysis)

    def test_risk_level_thresholds(self):
        self.assertEqual(graph_utils.risk_level(0.7), "high")
        self.assertEqual(graph_utils.risk_level(0.69), "medium")
        self.assertEqual(graph_utils.risk_level(0.4), "medium")
        self.assertEqual(graph_utils.risk_level(0.39), "low")
        self.assertEqual(graph_utils.risk_level(0.0), "none")


# ---------------------------------------------------------------------------
# Phase 1 & 2 Integration Tests (Full Cleaned Dataset + ML Risk Weighting)
# ---------------------------------------------------------------------------

class FullDatasetIntegrationTests(unittest.TestCase):
    def test_full_dataset_loads_and_validates_cleanly(self):
        import data_pipeline as dp

        tables = dp.load_cleaned_tables()
        sc = dp.load_full_supply_chain(data_dir=tables.data_dir, strict=True)
        self.assertTrue(sc.report.is_valid)
        self.assertEqual(len(sc.report.issues), 0)
        summary = graph_summary(sc.graph)
        self.assertEqual(summary["suppliers"], 80)
        self.assertEqual(summary["skus"], 500)
        self.assertEqual(summary["warehouses"], 5)
        self.assertEqual(summary["total_entities"], 585)
        self.assertEqual(summary["total_relationships"], 12956)

        kpis = dp.compute_executive_kpis(tables, sc.graph)
        self.assertEqual(kpis["total_pos"], 15000)
        self.assertEqual(kpis["total_shipments"], 20000)
        self.assertEqual(kpis["below_safety_stock_skus"], 32)
        self.assertGreater(len(kpis["top_bottlenecks"]), 0)

    def test_adapt_disruption_events_template_schema(self):
        import data_pipeline as dp

        template_df = pd.DataFrame([
            {
                "event_id": "EVT-1",
                "event_type": "port_strike",
                "affected_entity_type": "supplier",
                "affected_entity_id": "SUP-001",
                "location": "Germany",
                "start_date": "2024-10-01",
                "end_date": "2024-10-08",
                "duration_days": 7,
                "severity": 0.75,
                "source": "test",
                "is_synthetic_example": True,
            }
        ])
        adapted = dp.adapt_disruption_events_df(template_df)
        self.assertEqual(adapted.loc[0, "disruption_id"], "EVT-1")
        self.assertEqual(adapted.loc[0, "entity_id"], "SUP-001")
        self.assertEqual(adapted.loc[0, "disruption_type"], "port_strike")

    def test_ml_risk_weighting_and_inventory_ripple(self):
        import data_pipeline as dp

        tables = dp.load_cleaned_tables()
        sc = dp.load_full_supply_chain(data_dir=tables.data_dir, strict=True)
        d_event = disruption("SIM-T1", "SUP-001", 0.8, start="2024-11-01", days=14, kind="delay")

        base_res = ia.analyze_disruption(sc.graph, d_event, use_ml_weights=False)
        weighted = ia.apply_ml_risk_weights(
            sc.graph, node_risks={"SUP-001": 0.9}, edge_risks={}, risk_blend=0.5
        )
        ml_res = ia.analyze_disruption(weighted, d_event, use_ml_weights=True)

        self.assertEqual(base_res["summary"]["total_affected"], ml_res["summary"]["total_affected"])
        base_map = {a["entity_id"]: a["impact_score"] for a in base_res["affected_entities"]}
        ml_map = {a["entity_id"]: a["impact_score"] for a in ml_res["affected_entities"]}
        # Every affected downstream score with ML weighting should be >= baseline and <= 1.0
        for eid, b_score in base_map.items():
            self.assertGreaterEqual(ml_map[eid], b_score)
            self.assertLessEqual(ml_map[eid], 1.0)

        inv_ripple = ia.estimate_inventory_ripple(ml_res, tables.inventory, simulated_delay_days=14)
        self.assertFalse(inv_ripple.empty)
        self.assertIn("stockout_status", inv_ripple.columns)

    def test_ml_feature_engineering_is_leakage_safe(self):
        import data_pipeline as dp
        import ml_predictor as mlp

        forbidden = {
            "actual_delivery_date", "quantity_received", "po_status",
            "quality_accepted", "is_on_time", "days_late", "damage_reported",
        }
        self.assertTrue(forbidden.isdisjoint(set(mlp.PO_NUMERIC_FEATURES + mlp.PO_CATEGORICAL_FEATURES)))
        self.assertTrue(forbidden.isdisjoint(set(mlp.SHIPMENT_NUMERIC_FEATURES + mlp.SHIPMENT_CATEGORICAL_FEATURES)))

        if mlp.EVAL_RESULTS_PATH.is_file():
            tables = dp.load_cleaned_tables()
            sc = dp.load_full_supply_chain(data_dir=tables.data_dir, strict=True)
            po_bundle, ship_bundle, ev = mlp.load_model_bundles()
            self.assertIn("purchase_orders", ev)
            self.assertIn("shipments", ev)

            sample_pos = tables.purchase_orders.head(25)
            scored = mlp.score_purchase_orders(po_bundle, tables, sc.graph, po_subset=sample_pos)
            self.assertEqual(len(scored), 25)
            self.assertTrue((scored["delay_probability"] >= 0.0).all() and (scored["delay_probability"] <= 1.0).all())
            self.assertTrue((scored["expected_days_late"] >= 0.0).all())

            w_graph, node_risks, _, ml_dis = mlp.compute_dynamic_graph_risks(scored, sc.graph)
            self.assertGreater(len(node_risks), 0)
            self.assertEqual(w_graph.number_of_nodes(), sc.graph.number_of_nodes())


if __name__ == "__main__":
    unittest.main()
