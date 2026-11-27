# Supply chain dataset cleaning report

## Provenance
The repository README and generator identify the dataset as fully synthetic, generated for demonstration. It is not real-world operational data. The source generator uses seeded random-number generation (seed 2024), simulated suppliers/products/transactions, and generated dates spanning 2023-01-01 through 2024-12-31. Do not interpret the generated values as actual company performance.

## Cleaning performed
- Removed exact duplicate rows (none found in the inspected source tables).
- Checked missing values (none found). No imputation was needed.
- Standardized date columns to YYYY-MM-DD.
- Recomputed `days_late` as max(actual date − promised date, 0) and `is_on_time` from those dates in purchase orders and shipments.
- Retained all original purchase order and shipment columns, including IDs and date fields. Date columns use ISO `YYYY-MM-DD` format.
- Preserved remaining source columns and values.

## Tables
- `demand_history.csv`: 10500 rows, 8 columns.
- `inventory.csv`: 500 rows, 16 columns.
- `purchase_orders.csv`: 15000 rows, 15 columns.
- `shipments.csv`: 20000 rows, 14 columns.
- `suppliers.csv`: 80 rows, 12 columns.

## Supplier history
` supplier_historical_performance.csv` aggregates the complete purchase-order table by supplier: purchase order count, on-time rate, mean lateness, late order count, ordered/received quantities, accepted quality rate, and spend. The source `purchase_orders.csv` retains all transaction-level records used to calculate these values. The aggregation spans the full dataset period and is descriptive only; it includes outcomes from the same records and must not be used as a feature for a prediction made at order time. For leakage-safe model training, calculate supplier history using only records with an `order_date` earlier than each prediction cutoff (and only information available by that cutoff).

## Disruption history
No explicit disruption events, causes, dates, or event IDs appear in the supplied tables. `damage_reported` is an available shipment damage indicator; late delivery fields provide outcomes, not disruption causes. See `disruption_data_availability.csv`.

## Prediction targets
- Demand forecasting: `demand_history.demand_units` (forecast future demand by SKU, warehouse, and week).
- Supplier/order delivery: `purchase_orders.is_on_time` or `days_late` (avoid using post-delivery fields as predictors).
- Shipment delivery: `shipments.is_on_time` or `days_late`.
- Quality: `purchase_orders.quality_accepted`.
- Damage: `shipments.damage_reported` (proxy; depends on target availability and adequate positive examples).
- Supplier performance: use historical on-time rate or mean days late as outcomes from transaction-level records; supplier summary rates are already aggregate features and may leak target information.

### Label audit and temporal prediction note
The purchase-order labels were checked against dates: `is_on_time` equals `actual_delivery_date <= promised_delivery_date`; `days_late` equals the nonnegative calendar-day difference between actual and promised dates. All 15,000 purchase orders had both dates, and there were zero label mismatches. These fields are retained. Since actual delivery date and labels are known only after delivery, exclude them from predictors when predicting at order time; use them as training targets. The source data also has `po_status` and `quality_accepted`, but they are not delivery-time predictors.

### Purchase order and shipment linkage
Purchase orders have `po_id`; shipments have `shipment_id`. No common order/transaction ID is present, and the IDs do not match. `sku_id` and date fields overlap but do not establish a reliable transaction-level join. Keep the tables separate unless an authoritative crosswalk is provided.

### Disruption fields
No explicit disruption type, affected-entity event, event date, duration, or propagation fields are present. Do not create assumed disruption records.

## Feature types
The data dictionary labels each field numerical, categorical, boolean, or date. Identifiers such as `po_id`, `shipment_id`, `sku_id`, and `supplier_id` are categorical keys, not meaningful continuous measurements. Dates should be transformed into calendar or elapsed-time features for modeling.

## Team-requested additions
- `supplier_sku_warehouse_relationships.csv` contains combined PO and inventory-primary-supplier evidence, supporting PO counts, and first/last PO dates. Inventory-only links have count 0 and blank PO dates.
- `supplier_performance_as_of_date.csv` gives supplier snapshots for dates on which that supplier has an order. It only uses earlier POs whose actual delivery date is strictly before the cutoff, so delivery outcomes were observable at cutoff. Zero-history suppliers have blank rates/average and count 0. Use `as_of_date` features joined to orders at the same cutoff; retain source POs for auditable history.
- `disruption_events_template.csv` is an empty schema template. No events are fabricated. Required fields include event type, affected entity, location, dates, duration, severity and source.
- `data_dictionary.csv` now includes key roles, foreign-key references, prediction roles, and target columns. Candidate features still require a prediction-time availability check.
- `relationship_validation_report.csv` reports entity counts, relationships and unmatched identifiers. Warehouse names are treated as labels, not stable warehouse IDs.

## Data quality statistics
### Date coverage
- `purchase_orders.order_date`: 2023-01-01 to 2024-11-01
- `purchase_orders.promised_delivery_date`: 2023-01-06 to 2024-12-14
- `purchase_orders.actual_delivery_date`: 2023-01-05 to 2025-01-11
- `shipments.ship_date`: 2023-01-01 to 2024-12-31
- `shipments.promised_delivery_date`: 2023-01-02 to 2025-01-11
- `shipments.actual_delivery_date`: 2023-01-02 to 2025-01-16
- `demand_history.week_start`: 2023-01-01 to 2024-12-29

### Target and label distributions
- `purchase_orders.is_on_time`: `{False: 5584, True: 9416}`
- `purchase_orders.days_late`: `{'count': 15000.0, 'mean': 1.916, 'std': 4.073, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 2.0, 'max': 45.0}`
- `shipments.is_on_time`: `{False: 2884, True: 17116}`
- `shipments.days_late`: `{'count': 20000.0, 'mean': 0.574, 'std': 1.592, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 0.0, 'max': 7.0}`
- `shipments.damage_reported`: `{False: 19615, True: 385}`

### Label consistency
- Purchase order on-time and delay labels were recomputed from actual and promised dates; 15,000/15,000 records match (zero mismatches).
- Shipment labels were similarly recomputed from shipment actual and promised dates.
