"""
Inference logic — shared by the Streamlit app and any standalone script.
============================================================================
This module has NO Streamlit dependency on purpose: it's plain pandas/numpy,
so you can import and test it from a notebook, a script, or the app without
needing a Streamlit runtime. All app.py does is call these functions and
draw a UI around the result.

"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root — this file lives in app/, so repo root is one up)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "eta_model.pkl"
METRICS_PATH = ROOT / "model" / "metrics.json"

# ---------------------------------------------------------------------------
# Lookup tables + distributions — all real numbers from the training data
# ---------------------------------------------------------------------------
PRICE_PER_ITEM = {
    "Burger": 89.65, "Fried Chicken": 89.92, "Koshary": 90.16, "Pasta": 90.46,
    "Pizza": 89.79, "Salad": 90.11, "Sandwich": 89.76, "Shawarma": 90.27, "Sushi": 89.78,
}
CITY_COORDS = {
    "Alexandria": (31.2000, 29.9188), "Assiut": (27.1809, 31.1837),
    "Cairo": (30.0444, 31.2358), "Giza": (30.0132, 31.2089),
    "Mansoura": (31.0379, 31.3816), "Tanta": (30.7865, 30.9985),
    "Zagazig": (30.5877, 31.5021),
}
TRAFFIC_ORDINAL = {"Low": 1, "Medium": 2, "High": 3}
TRAFFIC_PROBS = {"Low": 0.458, "High": 0.292, "Medium": 0.251}
AVAILABILITY_PROBS = {"Online": 0.901, "Offline": 0.099}
DISTANCE_MEAN, DISTANCE_STD = 2.166, 1.039
DISTANCE_MIN, DISTANCE_MAX = 0.05, 5.3

VEHICLE_OPTIONS = ["Motorbike", "Car", "Bicycle"]
PAYMENT_OPTIONS = ["Cash", "Credit Card", "Wallet"]
ITEM_OPTIONS = ["Pizza", "Burger", "Sandwich", "Shawarma", "Fried Chicken",
                "Koshary", "Sushi", "Pasta", "Salad"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def point_at_distance(lat, lon, distance_km, bearing_deg, R=6371):
    lat1, lon1, brng = map(np.radians, [lat, lon, bearing_deg])
    d_R = distance_km / R
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_R) + np.cos(lat1) * np.sin(d_R) * np.cos(brng))
    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(d_R) * np.cos(lat1), np.cos(d_R) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


# ---------------------------------------------------------------------------
# Simulation of operational fields a real customer would never type in
# ---------------------------------------------------------------------------
def simulate_operational_fields(rng) -> dict:
    """rng: a numpy random Generator, e.g. np.random.default_rng()."""
    traffic_p = np.array(list(TRAFFIC_PROBS.values())); traffic_p /= traffic_p.sum()
    avail_p = np.array(list(AVAILABILITY_PROBS.values())); avail_p /= avail_p.sum()

    traffic_level = rng.choice(list(TRAFFIC_PROBS), p=traffic_p)
    driver_availability = rng.choice(list(AVAILABILITY_PROBS), p=avail_p)
    distance = float(np.clip(rng.normal(DISTANCE_MEAN, DISTANCE_STD), DISTANCE_MIN, DISTANCE_MAX))
    driver_distance = float(np.clip(rng.normal(1.0, 0.8), 0.05, 5.0))

    now = datetime.now()
    return {
        "traffic_level": str(traffic_level),
        "driver_availability": str(driver_availability),
        "delivery_distance_km": round(distance, 2),
        "driver_distance_km": round(driver_distance, 2),
        "order_hour": now.hour,
        "order_dayofweek": now.weekday(),
    }


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------
def build_features(order: dict, feature_columns: list) -> tuple[pd.DataFrame, float, dict]:
    """Turns one order dict into the exact feature row the model expects.

    Returns (X, total_price, locations) where locations has
    'restaurant' / 'customer' / 'driver' (lat, lon) tuples for mapping.
    """
    rest_lat, rest_lon = CITY_COORDS[order["city"]]
    cust_lat, cust_lon = point_at_distance(rest_lat, rest_lon, order["delivery_distance_km"], bearing_deg=135)
    driver_lat, driver_lon = point_at_distance(rest_lat, rest_lon, order["driver_distance_km"], bearing_deg=45)

    total_price = round(PRICE_PER_ITEM[order["item_name"]] * order["quantity"], 2)

    row = {
        "Quantity": order["quantity"], "Total_Price": total_price,
        "Restaurant_Lat": rest_lat, "Restaurant_Lon": rest_lon,
        "Customer_Lat": cust_lat, "Customer_Lon": cust_lon,
        "Driver_Lat": driver_lat, "Driver_Lon": driver_lon,
        "Delivery_Distance_km": order["delivery_distance_km"],
        "order_hour": order["order_hour"], "order_dayofweek": order["order_dayofweek"],
        "order_is_weekend": int(order["order_dayofweek"] in [4, 5]),
        "order_is_peak": int(order["order_hour"] in [12, 13, 19, 20, 21, 22]),
    }
    row["haversine_restaurant_customer_km"] = haversine(rest_lat, rest_lon, cust_lat, cust_lon)
    row["haversine_driver_restaurant_km"] = haversine(driver_lat, driver_lon, rest_lat, rest_lon)
    row["price_per_item"] = total_price / order["quantity"]
    row["traffic_x_distance"] = TRAFFIC_ORDINAL[order["traffic_level"]] * order["delivery_distance_km"]
    row["Total_Price_log"] = np.log1p(total_price)
    row["Delivery_Distance_km_log"] = np.log1p(order["delivery_distance_km"])
    row["price_per_item_log"] = np.log1p(row["price_per_item"])

    h = order["order_hour"]
    part = ("Late_Night" if h <= 5 else "Morning" if h <= 11 else
            "Afternoon" if h <= 16 else "Evening" if h <= 20 else "Night")

    for flag in [
        f"Item_Name_{order['item_name']}", f"Traffic_Level_{order['traffic_level']}",
        f"Driver_Vehicle_{order['driver_vehicle']}", f"Payment_Method_{order['payment_method']}",
        f"Driver_Availability_{order['driver_availability']}",
        f"order_part_of_day_{part}", f"City_{order['city']}",
    ]:
        row[flag] = 1

    X = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)
    locations = {
        "restaurant": (rest_lat, rest_lon), "customer": (cust_lat, cust_lon), "driver": (driver_lat, driver_lon),
    }
    return X, total_price, locations


# ---------------------------------------------------------------------------
# One-call convenience wrapper — handy for scripts/notebooks
# ---------------------------------------------------------------------------
def predict_order(model, feature_columns: list, customer_input: dict, seed=None) -> dict:
    """
    customer_input needs only: item_name, quantity, city, payment_method, driver_vehicle
    Everything operational is simulated. Pass `seed` for a reproducible result.
    """
    rng = np.random.default_rng(seed)
    order = {**customer_input, **simulate_operational_fields(rng)}
    X, total_price, locations = build_features(order, feature_columns)
    duration = float(model.predict(X)[0])
    return {
        "total_price_egp": total_price,
        "predicted_duration_minutes": round(duration, 1),
        "simulated_conditions": {k: order[k] for k in
            ["traffic_level", "driver_availability", "delivery_distance_km",
             "driver_distance_km", "order_hour", "order_dayofweek"]},
        "locations": locations,
    }


# ---------------------------------------------------------------------------
# Plain (uncached) loaders — app.py wraps these with @st.cache_resource /
# @st.cache_data; a plain script can just call them directly.
# ---------------------------------------------------------------------------
def load_model():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["feature_columns"], artifact.get("model_name", "Model")


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None