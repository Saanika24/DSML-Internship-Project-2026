"""
app.py
======
Phase 3 Interactive Streamlit Dashboard for AtmoGraph:
Supply Chain Ripple Effect Predictor (Infotact Solutions Capstone).

Pages
-----
1. Executive Overview & Supply Chain KPIs
2. Interactive Supply Chain Network & Ripple Simulator
3. ML Risk Predictions & Scenario Scoring
4. Model Evaluation & Data Insights (EDA)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from data_pipeline import compute_executive_kpis, load_cleaned_tables, load_full_supply_chain
from graph_utils import get_entity_ids_by_type, get_flow_graph, risk_level
from impact_analysis import (
    analyze_disruption,
    analyze_disruptions,
    estimate_inventory_ripple,
)
from ml_predictor import (
    MODELS_DIR,
    compute_dynamic_graph_risks,
    load_model_bundles,
    score_purchase_orders,
    score_shipments,
    train_and_save_models,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AtmoGraph | Supply Chain Ripple Effect Predictor",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {
    "disrupted": "#991B1B",  # deep crimson
    "high": "#EF4444",       # red
    "medium": "#F59E0B",     # amber
    "low": "#3B82F6",        # blue
    "none": "#10B981",       # emerald
}

TYPE_SHAPES = {
    "supplier": "diamond",
    "sku": "dot",
    "warehouse": "square",
    "factory": "triangle",
}


# ---------------------------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading supply chain datasets & building graph...")
def load_system_state():
    tables = load_cleaned_tables()
    sc = load_full_supply_chain(data_dir=tables.data_dir, strict=True)
    kpis = compute_executive_kpis(tables, sc.graph)
    return tables, sc, kpis


@st.cache_resource(show_spinner="Loading trained ML models (training if missing)...")
def load_or_train_models():
    eval_path = MODELS_DIR / "evaluation_results.json"
    if not eval_path.is_file():
        train_and_save_models(models_dir=MODELS_DIR)
    return load_model_bundles(MODELS_DIR)


@st.cache_data(show_spinner="Scoring active purchase orders & shipments...")
def get_scored_tables(_po_bundle, _ship_bundle, _tables, _graph):
    po_scored = score_purchase_orders(_po_bundle, _tables, _graph)
    ship_scored = score_shipments(_ship_bundle, _tables, _graph)
    weighted_graph, node_risks, expected_delays, ml_disruptions = compute_dynamic_graph_risks(
        po_scored, _graph, risk_blend=0.5
    )
    return po_scored, ship_scored, weighted_graph, node_risks, expected_delays, ml_disruptions


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def _build_ripple_subgraph(graph: nx.DiGraph, root_id: str, affected: List[Dict], max_nodes: int = 80) -> nx.DiGraph:
    """Extract a readable localized subgraph centered on the disrupted entity."""
    flow = get_flow_graph(graph)
    selected = {root_id}
    for item in affected[:max_nodes - 1]:
        selected.add(item["entity_id"])
    # Include immediate backup suppliers for affected SKUs so mitigation is visible
    for item in affected[:30]:
        eid = item["entity_id"]
        if graph.nodes[eid].get("entity_type") == "sku":
            for u, _, d in graph.in_edges(eid, data=True):
                if d.get("is_backup", False):
                    selected.add(u)
    sub = graph.subgraph(selected).copy()
    return sub


def render_pyvis_ripple(
    sub_graph: nx.DiGraph,
    root_id: str,
    impact_lookup: Dict[str, float],
    height: int = 560,
) -> str:
    """Build interactive PyVis HTML for the localized disruption ripple subgraph."""
    from pyvis.network import Network

    net = Network(
        height=f"{height}px",
        width="100%",
        directed=True,
        bgcolor="#0F172A",
        font_color="#E2E8F0",
    )
    net.barnes_hut(gravity=-4500, central_gravity=0.3, spring_length=140, spring_strength=0.04)

    for node_id, data in sub_graph.nodes(data=True):
        etype = data.get("entity_type", "unknown")
        name = data.get("entity_name") or node_id
        impact = impact_lookup.get(node_id, 0.0)
        if node_id == root_id:
            color = RISK_COLORS["disrupted"]
            size = 34
            status = "DISRUPTION ORIGIN"
        else:
            lvl = risk_level(impact)
            color = RISK_COLORS.get(lvl, "#64748B")
            size = 16 + int(impact * 18)
            status = f"Impact: {impact:.2%} ({lvl.upper()})"

        title = (
            f"<b>{node_id}</b> ({etype.upper()})<br>"
            f"Name: {name}<br>"
            f"Location: {data.get('location') or 'N/A'}<br>"
            f"{status}"
        )
        net.add_node(
            node_id,
            label=node_id,
            title=title,
            color=color,
            size=size,
            shape=TYPE_SHAPES.get(etype, "dot"),
        )

    for u, v, edge in sub_graph.edges(data=True):
        is_backup = bool(edge.get("is_backup", False))
        dep = float(edge.get("dependency_score", 0.0))
        u_imp = impact_lookup.get(u, 0.0)
        carrying = u_imp > 0 and not is_backup
        net.add_edge(
            u,
            v,
            value=max(1, dep * 5),
            dashes=is_backup,
            color="#F87171" if carrying else ("#38BDF8" if is_backup else "#475569"),
            title=f"{u} → {v} | dep={dep:.2f} | backup={is_backup}",
        )

    return net.generate_html()


def render_plotly_ripple(
    sub_graph: nx.DiGraph,
    root_id: str,
    impact_lookup: Dict[str, float],
) -> go.Figure:
    """Render a crisp tiered Plotly network diagram (Supplier -> SKU -> Warehouse)."""
    tier_x = {"supplier": 0.0, "factory": 1.0, "sku": 1.0, "warehouse": 2.0}
    grouped: Dict[float, List[str]] = {0.0: [], 1.0: [], 2.0: []}
    for n, d in sub_graph.nodes(data=True):
        x = tier_x.get(d.get("entity_type", "sku"), 1.0)
        grouped.setdefault(x, []).append(n)

    pos: Dict[str, tuple[float, float]] = {}
    for x, nodes in grouped.items():
        nodes_sorted = sorted(nodes, key=lambda nid: -impact_lookup.get(nid, 0.0))
        count = max(1, len(nodes_sorted))
        for idx, nid in enumerate(nodes_sorted):
            y = 1.0 - (idx + 0.5) / count
            pos[nid] = (x, y)

    edge_traces = []
    for u, v, d in sub_graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_backup = bool(d.get("is_backup", False))
        carrying = impact_lookup.get(u, 0.0) > 0 and not is_backup
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=1.8 if carrying else 0.9,
                    color="rgba(239, 68, 68, 0.65)" if carrying else "rgba(148, 163, 184, 0.35)",
                    dash="dash" if is_backup else "solid",
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )

    node_x, node_y, node_color, node_size, node_text, node_hover = [], [], [], [], [], []
    for n, d in sub_graph.nodes(data=True):
        x, y = pos[n]
        imp = impact_lookup.get(n, 0.0)
        lvl = "disrupted" if n == root_id else risk_level(imp)
        node_x.append(x)
        node_y.append(y)
        node_color.append(RISK_COLORS.get(lvl, "#64748B"))
        node_size.append(24 if n == root_id else 11 + int(imp * 14))
        node_text.append(n if (n == root_id or imp >= 0.25 or d.get("entity_type") == "warehouse") else "")
        node_hover.append(
            f"<b>{n}</b> ({d.get('entity_type', '').upper()})<br>"
            f"{d.get('entity_name', n)}<br>"
            f"Impact Score: {imp:.4f} ({lvl.upper()})"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(size=node_size, color=node_color, line=dict(width=1.2, color="#FFFFFF")),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Tiered Supply Chain Propagation (Supplier → SKU → Warehouse)",
        height=540,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["Suppliers", "SKUs", "Warehouses"],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
    )
    return fig


# ---------------------------------------------------------------------------
# Page 1: Executive Overview & Supply Chain KPIs
# ---------------------------------------------------------------------------

def render_page_overview(tables, sc, kpis, po_scored):
    st.title("🌐 AtmoGraph: Executive Supply Chain Overview")
    st.caption(
        "Unified view of the 585-node production network (80 Suppliers → 500 SKUs → 5 Warehouses), "
        "order & shipment reliability, inventory health, and top structural bottlenecks."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Suppliers / SKUs / WH", f"{kpis['total_suppliers']} / {kpis['total_skus']} / {kpis['total_warehouses']}")
    c2.metric("Active / Total POs", f"{kpis['pending_pos']} / {kpis['total_pos']:,}", f"Spend ${kpis['total_po_spend']/1e6:.2f}M")
    c3.metric("PO On-Time Rate", f"{kpis['po_on_time_rate']:.1%}", f"{kpis['late_pos']:,} late POs", delta_color="inverse")
    c4.metric("Shipment On-Time Rate", f"{kpis['shipment_on_time_rate']:.1%}", f"{kpis['damaged_shipments']} damaged", delta_color="inverse")
    c5.metric("Below Safety Stock", f"{kpis['below_safety_stock_skus']} SKUs", f"{kpis['below_reorder_point_skus']} below reorder", delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns([1.3, 1.0])
    bottlenecks_df = pd.DataFrame(kpis["top_bottlenecks"])

    with col_left:
        st.subheader("Top Vulnerable Supplier Bottlenecks")
        fig_bn = px.scatter(
            bottlenecks_df,
            x="downstream_nodes",
            y="observed_delay_rate",
            size="total_spend",
            color="risk_tier",
            hover_name="supplier_name",
            hover_data=["supplier_id", "country", "skus_supplied", "avg_days_late", "vulnerability_index"],
            labels={
                "downstream_nodes": "Downstream Network Reach (SKUs + Warehouses)",
                "observed_delay_rate": "Observed PO Delay Rate",
            },
            title="Supplier Risk Exposure: Network Reach vs. Delay Rate (Bubble = Spend)",
        )
        fig_bn.update_layout(height=410)
        st.plotly_chart(fig_bn, use_container_width=True)

    with col_right:
        st.subheader("Inventory Health by Warehouse")
        inv = tables.inventory.copy()
        inv["stock_status"] = np.where(
            inv["current_stock_qty"] < inv["safety_stock_qty"],
            "Below Safety Stock",
            np.where(inv["current_stock_qty"] <= inv["reorder_point_qty"], "Below Reorder Point", "Healthy"),
        )
        wh_status = inv.groupby(["warehouse", "stock_status"]).size().reset_index(name="sku_count")
        fig_wh = px.bar(
            wh_status,
            x="warehouse",
            y="sku_count",
            color="stock_status",
            barmode="stack",
            color_discrete_map={
                "Below Safety Stock": "#EF4444",
                "Below Reorder Point": "#F59E0B",
                "Healthy": "#10B981",
            },
            title="SKU Stock Health Across Warehouses",
        )
        fig_wh.update_layout(height=410)
        st.plotly_chart(fig_wh, use_container_width=True)

    st.subheader("Top 15 Bottleneck Suppliers (Ranked by Vulnerability Index)")
    st.dataframe(
        bottlenecks_df[
            [
                "supplier_id",
                "supplier_name",
                "country",
                "risk_tier",
                "skus_supplied",
                "downstream_nodes",
                "po_count",
                "observed_delay_rate",
                "avg_days_late",
                "vulnerability_index",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    high_risk_pos = int((po_scored["risk_level"] == "high").sum())
    med_risk_pos = int((po_scored["risk_level"] == "medium").sum())
    st.info(
        f"**ML Risk Snapshot:** Across {len(po_scored):,} scored POs, **{high_risk_pos:,}** are flagged **HIGH risk** "
        f"(p >= 0.70) and **{med_risk_pos:,}** are **MEDIUM risk** (0.40 <= p < 0.70). "
        "Switch to **Network & Ripple Simulator** or **ML Risk Predictions** for interactive drill-downs."
    )


# ---------------------------------------------------------------------------
# Page 2: Interactive Supply Chain Network & Ripple Simulator
# ---------------------------------------------------------------------------

def render_page_simulator(tables, sc, weighted_graph):
    st.title("🕸️ Interactive Supply Chain Network & Ripple Simulator")
    st.caption(
        "Select any Supplier, SKU, or Warehouse, inject a disruption scenario, and trace the cascading "
        "impact across downstream SKUs, warehouses, active/recent purchase orders, and inventory buffers."
    )

    with st.sidebar:
        st.subheader("Ripple Simulation Controls")
        entity_type = st.selectbox("Disrupted Entity Type", ["supplier", "sku", "warehouse"])
        candidates = get_entity_ids_by_type(sc.graph, entity_type)
        name_map = {
            eid: f"{eid} — {sc.graph.nodes[eid].get('entity_name', eid)}" for eid in candidates
        }
        selected_entity = st.selectbox(
            "Select Target Entity",
            candidates,
            format_func=lambda x: name_map.get(x, x),
        )
        disruption_type = st.selectbox(
            "Disruption Event Type",
            ["supplier_delay", "flood", "port_strike", "capacity_drop", "customs_hold", "fire"],
        )
        severity = st.slider("Disruption Severity (Capacity Drop)", 0.10, 1.00, 0.80, 0.05)
        delay_days = st.slider("Simulated Delay Duration (Days)", 1, 45, 14, 1)
        use_ml_weights = st.checkbox(
            "Amplify with ML Dynamic Risk Weights",
            value=True,
            help="Scales structural edge dependency_score using ML-predicted delay probabilities.",
        )
        viz_engine = st.radio("Graph Renderer", ["PyVis Interactive Physics", "Plotly Tiered Layout"], horizontal=True)

    active_graph = weighted_graph if use_ml_weights else sc.graph
    disruption_event = {
        "disruption_id": "SIM-001",
        "entity_id": selected_entity,
        "disruption_type": disruption_type,
        "severity": float(severity),
        "location": active_graph.nodes[selected_entity].get("location"),
        "start_date": date.today().isoformat(),
        "duration_days": int(delay_days),
    }

    result = analyze_disruption(
        active_graph,
        disruption_event,
        max_paths=500,
        min_impact=0.0,
        use_ml_weights=use_ml_weights,
    )
    affected = result["affected_entities"]
    impact_lookup = {selected_entity: float(severity)}
    for item in affected:
        impact_lookup[item["entity_id"]] = float(item["impact_score"])

    inv_ripple = estimate_inventory_ripple(result, tables.inventory, simulated_delay_days=delay_days)
    stockouts = int((inv_ripple["stockout_status"] == "Stockout").sum()) if not inv_ripple.empty else 0
    safety_breaches = int((inv_ripple["stockout_status"] == "Safety Breach").sum()) if not inv_ripple.empty else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Affected Nodes", result["summary"]["total_affected"])
    m2.metric("Affected SKUs", result["summary"]["affected_by_type"].get("sku", 0))
    m3.metric("Affected Warehouses", result["summary"]["affected_by_type"].get("warehouse", 0))
    m4.metric("Highest Downstream Risk", result["summary"]["highest_risk_level"].upper())
    m5.metric("Inventory Shock Alerts", f"{stockouts} Stockout / {safety_breaches} Safety")

    sub = _build_ripple_subgraph(active_graph, selected_entity, affected, max_nodes=75)
    st.caption(
        f"Displaying localized ripple subgraph around **{selected_entity}** "
        f"({sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges)."
    )
    if viz_engine == "PyVis Interactive Physics":
        html_str = render_pyvis_ripple(sub, selected_entity, impact_lookup, height=540)
        components.html(html_str, height=560, scrolling=False)
    else:
        fig_net = render_plotly_ripple(sub, selected_entity, impact_lookup)
        st.plotly_chart(fig_net, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Downstream Impacted Nodes", "Inventory Stockout Ripple", "Propagation Paths", "Affected Purchase Orders"]
    )
    with tab1:
        if affected:
            st.dataframe(pd.DataFrame(affected), use_container_width=True, hide_index=True)
        else:
            st.info("No downstream entities affected from this node.")
    with tab2:
        if not inv_ripple.empty:
            st.dataframe(inv_ripple, use_container_width=True, hide_index=True)
        else:
            st.info("No SKU nodes impacted in this simulation.")
    with tab3:
        paths = result["propagation_paths"]
        if paths:
            path_df = pd.DataFrame({"hops": [len(p) - 1 for p in paths], "propagation_path": [" → ".join(p) for p in paths]})
            st.dataframe(path_df, use_container_width=True, hide_index=True)
        else:
            st.info("No propagation paths generated.")
    with tab4:
        affected_ids = set(impact_lookup.keys())
        po = tables.purchase_orders
        if entity_type == "supplier":
            po_hits = po[po["supplier_id"] == selected_entity]
        elif entity_type == "sku":
            po_hits = po[po["sku_id"] == selected_entity]
        else:
            po_hits = po[po["warehouse"] == selected_entity]
        po_hits = po_hits.sort_values("order_date", ascending=False).head(100)
        st.caption(f"Showing latest {len(po_hits)} historical/active POs linked to {selected_entity} (affected downstream nodes: {len(affected_ids)}).")
        st.dataframe(
            po_hits[["po_id", "supplier_id", "sku_id", "warehouse", "order_date", "promised_delivery_date", "quantity_ordered", "total_cost", "po_status", "is_on_time", "days_late"]],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page 3: ML Risk Predictions & Scenario Scoring
# ---------------------------------------------------------------------------

def render_page_predictions(tables, sc, po_bundle, po_scored, ship_scored, weighted_graph, ml_disruptions):
    st.title("🤖 ML Risk Predictions & Live Scenario Scoring")
    st.caption(
        "Real-time disruption probability and expected delay scoring powered by trained gradient-boosted "
        "classification & regression models, feeding dynamic risk weights into the ripple engine."
    )

    tab_po, tab_ship, tab_whatif, tab_network = st.tabs(
        ["Purchase Order Risk Queue", "Shipment Risk Queue", "Live What-If Scenario Scorer", "Network-Wide ML Exposure"]
    )

    with tab_po:
        f1, f2, f3 = st.columns(3)
        wh_filter = f1.multiselect("Warehouse", sorted(po_scored["warehouse"].unique()), default=[])
        risk_filter = f2.multiselect("Risk Level", ["high", "medium", "low", "none"], default=["high", "medium"])
        status_only = f3.checkbox("Show Pending / Latest 500 POs Only", value=False)

        view = po_scored.copy()
        if wh_filter:
            view = view[view["warehouse"].isin(wh_filter)]
        if risk_filter:
            view = view[view["risk_level"].isin(risk_filter)]
        if status_only:
            pending = view[view["po_status"] == "pending"]
            view = pending if not pending.empty else view.sort_values("order_date", ascending=False).head(500)

        st.metric("Matching Purchase Orders", f"{len(view):,}")
        st.dataframe(
            view.head(300)[
                [
                    "po_id",
                    "supplier_id",
                    "sku_id",
                    "warehouse",
                    "order_date",
                    "quantity_ordered",
                    "promised_lead_days",
                    "delay_probability",
                    "expected_days_late",
                    "risk_level",
                    "recommended_action",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("High-Risk Suppliers Leaderboard (ML Aggregated)")
        sup_risk = (
            po_scored.groupby("supplier_id")
            .agg(
                scored_pos=("po_id", "count"),
                mean_delay_prob=("delay_probability", "mean"),
                max_delay_prob=("delay_probability", "max"),
                mean_expected_days_late=("expected_days_late", "mean"),
            )
            .reset_index()
            .merge(tables.suppliers[["supplier_id", "supplier_name", "country", "risk_tier"]], on="supplier_id", how="left")
            .sort_values("mean_delay_prob", ascending=False)
        )
        st.dataframe(sup_risk.head(20), use_container_width=True, hide_index=True)

    with tab_ship:
        s1, s2 = st.columns(2)
        carriers = s1.multiselect("Carrier", sorted(ship_scored["carrier"].unique()), default=[])
        s_risk = s2.multiselect("Shipment Risk Level", ["high", "medium", "low", "none"], default=["high", "medium"])
        sview = ship_scored.copy()
        if carriers:
            sview = sview[sview["carrier"].isin(carriers)]
        if s_risk:
            sview = sview[sview["risk_level"].isin(s_risk)]
        st.dataframe(sview.head(300), use_container_width=True, hide_index=True)

    with tab_whatif:
        st.subheader("Score a New / Hypothetical Purchase Order")
        with st.form("whatif_po_form"):
            w1, w2, w3 = st.columns(3)
            sup_id = w1.selectbox("Supplier", sorted(tables.suppliers["supplier_id"].unique()))
            sku_id = w2.selectbox("SKU", sorted(tables.inventory["sku_id"].unique()))
            wh_id = w3.selectbox("Destination Warehouse", sorted(tables.purchase_orders["warehouse"].unique()))

            w4, w5, w6 = st.columns(3)
            qty = w4.number_input("Quantity Ordered", min_value=1, max_value=5000, value=120)
            lead_days = w5.slider("Promised Lead Time (Days)", min_value=2, max_value=60, value=14)
            order_dt = w6.date_input("Order Date", value=date(2024, 11, 1))
            submitted = st.form_submit_button("Predict Disruption Risk")

        if submitted:
            unit_cost = float(
                tables.inventory.loc[tables.inventory["sku_id"] == sku_id, "unit_cost"].iloc[0]
            )
            scenario_po = pd.DataFrame(
                [
                    {
                        "po_id": "PO-WHATIF-001",
                        "supplier_id": sup_id,
                        "sku_id": sku_id,
                        "warehouse": wh_id,
                        "order_date": order_dt.isoformat(),
                        "promised_delivery_date": (order_dt + timedelta(days=int(lead_days))).isoformat(),
                        "quantity_ordered": int(qty),
                        "unit_cost": unit_cost,
                        "total_cost": round(unit_cost * int(qty), 2),
                        "po_status": "pending",
                    }
                ]
            )
            scored_one = score_purchase_orders(po_bundle, tables, sc.graph, po_subset=scenario_po).iloc[0]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Predicted Delay Probability", f"{scored_one['delay_probability']:.1%}")
            r2.metric("Expected Delay Duration", f"{scored_one['expected_days_late']:.1f} days")
            r3.metric("Predicted Risk Tier", scored_one["risk_level"].upper())
            r4.metric("Graph Edge Dependency", f"{scored_one['edge_dependency_score']:.2%}")
            st.warning(f"**Recommended Mitigation:** {scored_one['recommended_action']}")

    with tab_network:
        st.subheader("ML-Driven Baseline Network Exposure")
        st.caption(
            f"Propagating {len(ml_disruptions)} supplier nodes with predicted recent delay probability >= 0.40 "
            "through the risk-weighted supply chain graph."
        )
        if ml_disruptions:
            net_analysis = analyze_disruptions(
                weighted_graph,
                ml_disruptions,
                max_paths=200,
                use_ml_weights=True,
            )
            comb_df = pd.DataFrame(net_analysis["combined_impact"])
            st.dataframe(comb_df.head(50), use_container_width=True, hide_index=True)
        else:
            st.success("No suppliers exceed the 0.40 ML risk threshold in the recent window.")


# ---------------------------------------------------------------------------
# Page 4: Model Evaluation & Data Insights (EDA)
# ---------------------------------------------------------------------------

def render_page_evaluation(tables, evaluation):
    st.title("📊 Model Evaluation & Exploratory Data Insights")
    st.caption(
        "Technical review panel: chronological 80/20 holdout evaluation (zero future leakage), "
        "multi-model comparison, confusion matrices, ROC curves, SHAP feature drivers, and EDA."
    )

    domain = st.radio(
        "Select Modeling Pipeline",
        ["purchase_orders", "shipments"],
        format_func=lambda x: "Purchase Orders (Delay & Severity)" if x == "purchase_orders" else "Shipments (Transit Delay & Severity)",
        horizontal=True,
    )
    ev = evaluation[domain]

    st.info(
        f"**Chronological Split:** Train = {ev['train_rows']:,} rows ({ev['train_date_range'][0]} → {ev['train_date_range'][1]}) | "
        f"Test = {ev['test_rows']:,} rows ({ev['test_date_range'][0]} → {ev['test_date_range'][1]}) | "
        f"Best Classifier = **{ev['best_classifier']}** | Best Regressor = **{ev['best_regressor']}**"
    )

    cls_rows = []
    for name, m in ev["classification_models"].items():
        cls_rows.append(
            {
                "Model": name,
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
                "F1": m["f1"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "Accuracy": m["accuracy"],
            }
        )
    reg_rows = []
    for name, m in ev["regression_models"].items():
        reg_rows.append({"Model": name, "RMSE (days)": m["rmse"], "MAE (days)": m["mae"], "R2": m["r2"]})

    c_tbl1, c_tbl2 = st.columns(2)
    with c_tbl1:
        st.subheader("Classification Comparison")
        st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)
    with c_tbl2:
        st.subheader("Regression Comparison (Days Late)")
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    c_plot1, c_plot2 = st.columns(2)
    with c_plot1:
        fig_roc = go.Figure()
        for name, m in ev["classification_models"].items():
            rc = m["roc_curve"]
            fig_roc.add_trace(
                go.Scatter(
                    x=rc["fpr"],
                    y=rc["tpr"],
                    mode="lines",
                    name=f"{name} (AUC={m['roc_auc']:.3f})",
                )
            )
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random")
        )
        fig_roc.update_layout(
            title="ROC Curves (Holdout Test Set)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with c_plot2:
        selected_m = st.selectbox("Confusion Matrix Model", list(ev["classification_models"].keys()), index=list(ev["classification_models"].keys()).index(ev["best_classifier"]))
        cm = ev["classification_models"][selected_m]["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted On-Time", "Predicted Delayed"],
            y=["Actual On-Time", "Actual Delayed"],
            color_continuous_scale="Blues",
            title=f"Confusion Matrix — {selected_m}",
        )
        fig_cm.update_layout(height=360)
        st.plotly_chart(fig_cm, use_container_width=True)

    c_imp1, c_imp2 = st.columns(2)
    with c_imp1:
        shap_df = pd.DataFrame(ev["shap_summary"][:15])
        if not shap_df.empty:
            fig_shap = px.bar(
                shap_df.sort_values("mean_abs_shap"),
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title=f"Top 15 SHAP Feature Drivers ({ev['best_classifier']})",
            )
            fig_shap.update_layout(height=440)
            st.plotly_chart(fig_shap, use_container_width=True)
    with c_imp2:
        fi_df = pd.DataFrame(ev["classification_models"][ev["best_classifier"]]["feature_importance"][:15])
        if not fi_df.empty:
            fig_fi = px.bar(
                fi_df.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top 15 Tree Feature Importances ({ev['best_classifier']})",
            )
            fig_fi.update_layout(height=440)
            st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()
    st.subheader("Exploratory Data Analysis (EDA)")
    eda1, eda2 = st.columns(2)
    with eda1:
        po = tables.purchase_orders
        fig_dl = px.histogram(
            po[po["days_late"] > 0],
            x="days_late",
            nbins=30,
            title="Purchase Order Delay Severity Distribution (Late POs Only)",
            labels={"days_late": "Days Beyond Promised Delivery"},
        )
        st.plotly_chart(fig_dl, use_container_width=True)
    with eda2:
        ship = tables.shipments.copy()
        ship["is_delayed"] = (~ship["is_on_time"]).astype(int)
        carrier_rates = ship.groupby("carrier", as_index=False)["is_delayed"].mean().sort_values("is_delayed", ascending=False)
        fig_car = px.bar(
            carrier_rates,
            x="carrier",
            y="is_delayed",
            title="Shipment Delay Rate by Carrier",
            labels={"is_delayed": "Delay Rate"},
        )
        st.plotly_chart(fig_car, use_container_width=True)

    dem = tables.demand_history.groupby(["week_start", "category"], as_index=False)["demand_units"].sum()
    fig_dem = px.line(
        dem,
        x="week_start",
        y="demand_units",
        color="category",
        title="Weekly Demand History by Product Category (2023–2024)",
    )
    st.plotly_chart(fig_dem, use_container_width=True)


# ---------------------------------------------------------------------------
# Main App Router
# ---------------------------------------------------------------------------

def main():
    tables, sc, kpis = load_system_state()
    po_bundle, ship_bundle, evaluation = load_or_train_models()
    po_scored, ship_scored, weighted_graph, _, _, ml_disruptions = get_scored_tables(
        po_bundle, ship_bundle, tables, sc.graph
    )

    with st.sidebar:
        st.title("🌐 AtmoGraph")
        st.caption("Infotact Solutions Capstone")
        page = st.radio(
            "Navigation",
            [
                "1. Executive Overview & KPIs",
                "2. Network & Ripple Simulator",
                "3. ML Risk & Scenario Scoring",
                "4. Model Evaluation & EDA",
            ],
        )
        st.divider()
        st.caption(
            f"Graph: {sc.graph.number_of_nodes()} nodes · {sc.graph.number_of_edges():,} edges\n"
            f"Best PO Classifier: {evaluation['purchase_orders']['best_classifier']} "
            f"(AUC {evaluation['purchase_orders']['classification_models'][evaluation['purchase_orders']['best_classifier']]['roc_auc']:.3f})"
        )

    if page.startswith("1."):
        render_page_overview(tables, sc, kpis, po_scored)
    elif page.startswith("2."):
        render_page_simulator(tables, sc, weighted_graph)
    elif page.startswith("3."):
        render_page_predictions(tables, sc, po_bundle, po_scored, ship_scored, weighted_graph, ml_disruptions)
    else:
        render_page_evaluation(tables, evaluation)


if __name__ == "__main__":
    main()
