from pathlib import Path
import pandas as pd

from src.prediction.predict import predict_new_purchase_orders

ROOT = Path(__file__).resolve().parents[2]

historical_orders = pd.read_csv(
    ROOT / "data" / "raw" / "purchase_orders.csv"
)

historical_orders["order_date"] = pd.to_datetime(
    historical_orders["order_date"]
)

# Create one new, unlabeled order using an existing supplier and SKU.
sample = historical_orders.iloc[-1]

new_order_date = historical_orders["order_date"].max() + pd.Timedelta(days=1)

new_purchase_orders = pd.DataFrame([{
    "supplier_id": sample["supplier_id"],
    "sku_id": sample["sku_id"],
    "warehouse": sample["warehouse"],
    "order_date": new_order_date,
    "promised_delivery_date": new_order_date + pd.Timedelta(days=14),
    "quantity_ordered": sample["quantity_ordered"],
    "unit_cost": sample["unit_cost"],
}])

print("\nNew unlabeled purchase order:")
print(new_purchase_orders.to_string(index=False))

print("\nGenerating predictions...")

results = predict_new_purchase_orders(
    new_orders=new_purchase_orders,
    historical_orders=historical_orders,
    model_dir="models",
    regressor_dir="models_regression_experiment",
)

print("\nPrediction result:")
print(results.to_string(index=False))

print("\nInference test completed successfully.")
