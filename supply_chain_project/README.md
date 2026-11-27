# AtmoGraph: Supply Chain Ripple Effect Predictor

**Infotact Solutions Capstone Project** — Unified supply-chain data engineering, graph-based disruption propagation, leakage-safe machine learning risk prediction, and an interactive multi-page Streamlit decision dashboard.

---

## 1. Architecture Overview

```text
c:\intern\                                  # Cleaned production CSV layer
├── suppliers.csv                           # 80 suppliers (country, category, lead time, risk tier)
├── inventory.csv                           # 500 SKUs (stock, safety stock, reorder point, days of stock)
├── purchase_orders.csv                     # 15,000 POs (2023-01-01 to 2024-11-01)
├── shipments.csv                           # 20,000 shipments (2023-01-01 to 2024-12-31)
├── demand_history.csv                      # 10,500 weekly SKU-warehouse demand records
├── supplier_performance_as_of_date.csv     # 13,119 leakage-safe order-date supplier snapshots
└── supplier_sku_warehouse_relationships.csv

c:\intern\supply_chain_project\             # Unified Graph + ML + Dashboard Engine
├── data_pipeline.py                        # Phase 1: Cleaned-CSV -> Graph ingestion adapter & KPIs
├── graph_utils.py                          # Entity/relationship schema, validation report, graph queries
├── graph_builder.py                        # NetworkX DiGraph builder & validator (legacy + PO modes)
├── impact_analysis.py                      # Disruption propagation, backup mitigation, ML risk weights & inventory ripple
├── neo4j_sync.py                           # Parameterized Cypher export & Neo4j sync helpers
├── ml_predictor.py                         # Phase 2: Feature engineering, model training, SHAP & risk scoring
├── models\                                 # Saved joblib model bundles & evaluation_results.json
├── app.py                                  # Phase 3: 4-page interactive Streamlit dashboard
├── test_supply_chain.py                    # 105 unit & integration tests
└── requirements.txt
```

### Graph Topology
- **Production Network (`Supplier → SKU → Warehouse`):** 585 nodes (`80` suppliers, `500` SKUs, `5` warehouses) and `12,956` directed dependency edges (`dependency_score` computed from ordered volume shares).
- **Backward Compatibility:** The legacy `Supplier → Factory → Warehouse` schema and sample fixtures under `data/` remain fully supported and tested.

---

## 2. Setup & Installation

```bash
cd c:\intern\supply_chain_project
pip install -r requirements.txt
```

### Run Tests (Unit + Full-Dataset Integration)
```bash
python -m unittest -v
# or
pytest -v
```

### Train / Refresh ML Models
```bash
python ml_predictor.py
```
This trains and evaluates all classification and regression models chronologically, saves `models/po_model_bundle.joblib`, `models/shipment_model_bundle.joblib`, and `models/evaluation_results.json`.

### Launch the Interactive Dashboard
```bash
python -m streamlit run app.py
```

---

## 3. ML Methodology & Leakage Prevention

1. **Disjoint Transaction Keys Handled Separately:** `purchase_orders.csv` (`po_id`) and `shipments.csv` (`shipment_id`) share zero common transaction IDs (`relationship_validation_report.csv`), so separate domain models are trained for Purchase Order delays and Shipment transit delays.
2. **Strict Leakage-Safe Features:**
   - Excludes post-delivery columns (`actual_delivery_date`, `quantity_received`, `po_status`, `quality_accepted`, `is_on_time`, `days_late`, `damage_reported`) and full-period `supplier_historical_performance.csv`.
   - Uses `supplier_performance_as_of_date.csv` matched on `(supplier_id, order_date == as_of_date)` alongside strictly causal rolling-5 supplier windows (`actual_delivery_date < current_order_date`), causal SKU quantity anomaly z-scores, inventory buffer ratios, and NetworkX graph structural/centrality metrics (`degree`, `downstream_count`, `edge_dependency_score`).
   - Uses a **chronological 80/20 train/test split** (earliest 80% train, latest 20% holdout test).
3. **Evaluated Models (Holdout Test Metrics):**
   - **Purchase Orders (Train = 12,000 | Test = 3,000):**
     - Classification (`is_delayed`): **GradientBoosting** (`ROC-AUC = 0.5961`, `PR-AUC = 0.4299`, `F1 = 0.5319`), compared against `LogisticRegression` (`0.5920`), `XGBoost` (`0.5897`), and `RandomForest` (`0.5891`).
     - Regression (`days_late`): **GradientBoosting** (`RMSE = 3.4540`, `MAE = 2.2826`, `R² = 0.1083`), compared against `Ridge` (`RMSE = 3.4701`), `XGBoost` (`3.4818`), and `RandomForest` (`3.5026`).
     - Top SHAP Drivers: `lead_time_variability`, `promised_lead_days`, `prior_average_days_late`, `avg_lead_time_days`, `rolling_5_lead_time_std`.
   - **Shipments (Train = 16,000 | Test = 4,000):**
     - Classification (`is_delayed`): **LogisticRegression** (`ROC-AUC = 0.6023`, `F1 = 0.2730`), closely matched by **GradientBoosting** (`ROC-AUC = 0.6021`, `F1 = 0.2748`), `RandomForest` (`0.5988`), and `XGBoost` (`0.5952`).
     - Regression (`days_late`): **Ridge** (`RMSE = 1.5523`, `MAE = 0.9200`, `R² = 0.0100`).
     - Top SHAP Drivers: `carrier_FedEx`, `carrier_Local Fleet`, `carrier_prior_delay_rate`, `carrier_UPS`, `carrier_prior_avg_days_late`.
4. **Dynamic Risk Integration:** Predicted delay probabilities feed into `impact_analysis.apply_ml_risk_weights()` and `generate_ml_risk_disruptions()` to dynamically weight network propagation and estimate downstream inventory stockout exposure.

---

## 4. Streamlit Dashboard Guide (`app.py`)

1. **Page 1 — Executive Overview & KPIs:** Network totals, PO & shipment on-time rates, inventory safety-stock alerts, warehouse stock breakdown, and top bottleneck suppliers ranked by Vulnerability Index.
2. **Page 2 — Network & Ripple Simulator:** Select any Supplier, SKU, or Warehouse, inject disruption severity and delay days, toggle ML dynamic risk weights, and explore interactive PyVis / Plotly ripple graphs alongside SKU stockout alerts and propagation paths.
3. **Page 3 — ML Risk & Scenario Scoring:** Filterable PO & Shipment risk queues with mitigation recommendations, high-risk supplier leaderboard, live "What-If" PO risk scorer, and network-wide baseline ML exposure.
4. **Page 4 — Model Evaluation & EDA:** Interactive model comparison tables, ROC curves, confusion matrices, SHAP & tree feature importances, delay histograms, carrier reliability, and weekly demand trends.
