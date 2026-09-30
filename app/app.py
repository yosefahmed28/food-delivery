"""
Food Delivery ETA Predictor — Streamlit app
==============================================
Run from the repo root:
    streamlit run app/streamlit_app.py

All prediction logic lives in inference.py (shared with any standalone
script/notebook); this file is UI only. Only asks the customer for what a
real customer would know (item, quantity, city, payment method, vehicle).
Everything operational — traffic, driver availability, distances, order
time — is simulated from the real training data's own distributions.
"""

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from inference import (
    CITY_COORDS, VEHICLE_OPTIONS, PAYMENT_OPTIONS, ITEM_OPTIONS,
    simulate_operational_fields, build_features, load_model, load_metrics,
)

# ---------------------------------------------------------------------------
# Cached loaders — thin wrappers around inference.py's plain functions
# ---------------------------------------------------------------------------
cached_load_model = st.cache_resource(load_model)
cached_load_metrics = st.cache_data(load_metrics)

# ---------------------------------------------------------------------------
# Page setup + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Delivery ETA Predictor", page_icon="🛵", layout="centered")

st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at top left, #1a1f2b 0%, #0E1117 55%); }
    div[data-testid="stMetric"] {
        background-color: #1C2128;
        border: 1px solid #2A313C;
        border-radius: 12px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricValue"] { color: #FF6B4A; }
    .hero {
        text-align: center;
        padding: 6px 0 18px 0;
    }
    .hero h1 { margin-bottom: 0px; }
    .hero p { color: #9CA3AF; margin-top: 4px; }
    .result-card {
        background: linear-gradient(135deg, #1C2128 0%, #262D38 100%);
        border: 1px solid #333B47;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 10px;
    }
    .result-card .big-number { font-size: 52px; font-weight: 700; color: #FF6B4A; }
    .result-card .label { color: #9CA3AF; font-size: 14px; letter-spacing: 1px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛵 Delivery ETA Predictor</h1>
    <p>Tell us what you're ordering — we'll estimate the price and delivery time.</p>
</div>
""", unsafe_allow_html=True)

model, feature_columns, model_name = cached_load_model()
metrics = cached_load_metrics()

if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model_name)
    c2.metric("R²", f"{metrics['r2']:.2f}")
    c3.metric("Avg. error", f"± {metrics['mae']:.1f} min")

st.divider()

# ---------------------------------------------------------------------------
# Order form — only fields a real customer would actually provide
# ---------------------------------------------------------------------------
with st.form("order_form"):
    st.subheader("Your order")
    col1, col2 = st.columns(2)
    with col1:
        item_name = st.selectbox("Item", ITEM_OPTIONS)
        quantity = st.number_input("Quantity", min_value=1, max_value=5, value=2)
        city = st.selectbox("City", list(CITY_COORDS.keys()))
    with col2:
        payment_method = st.selectbox("Payment method", PAYMENT_OPTIONS)
        driver_vehicle = st.selectbox("Driver vehicle", VEHICLE_OPTIONS)

    submitted = st.form_submit_button("Estimate my order 🚀", use_container_width=True)

if submitted:
    rng = np.random.default_rng()  # fresh simulation every submit (new customer)
    customer_input = {
        "item_name": item_name, "quantity": quantity, "city": city,
        "payment_method": payment_method, "driver_vehicle": driver_vehicle,
    }
    order = {**customer_input, **simulate_operational_fields(rng)}   # change every time
    X, total_price, locations = build_features(order, feature_columns)
    predicted_minutes = float(model.predict(X)[0])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated delivery time</div>
            <div class="big-number">{predicted_minutes:.0f}</div>
            <div class="label">minutes</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Total price</div>
            <div class="big-number">{total_price:.0f}</div>
            <div class="label">EGP</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("See simulated delivery conditions"):
        st.caption(
            "Traffic, driver status, and distance aren't things you'd type in yourself — "
            "in a real system they'd come from the live dispatch/traffic backend. Here "
            "they're simulated from the same distributions seen in the real training data."
        )
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Traffic", order["traffic_level"])
        cc2.metric("Driver status", order["driver_availability"])
        cc3.metric("Distance", f"{order['delivery_distance_km']} km")

        map_df = pd.DataFrame({
            "lat": [locations["restaurant"][0], locations["customer"][0], locations["driver"][0]],
            "lon": [locations["restaurant"][1], locations["customer"][1], locations["driver"][1]],
            "label": ["Restaurant", "Customer", "Driver"],
            "color": [[255, 107, 74], [76, 175, 80], [66, 165, 245]],  # orange, green, blue
        })

        legend_cols = st.columns(3)
        swatches = ["🟧 Restaurant", "🟩 Customer", "🟦 Driver"]
        for col, text in zip(legend_cols, swatches):
            col.markdown(f"<div style='text-align:center; color:#E6E6E6;'>{text}</div>", unsafe_allow_html=True)

        view_state = pdk.ViewState(
            latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(),
            zoom=12.5, pitch=0,
        )
        scatter_layer = pdk.Layer(
            "ScatterplotLayer", data=map_df,
            get_position="[lon, lat]", get_fill_color="color",
            get_radius=80, radius_min_pixels=8, radius_max_pixels=20,
            pickable=True,
        )
        text_layer = pdk.Layer(
            "TextLayer", data=map_df,
            get_position="[lon, lat]", get_text="label",
            get_color=[230, 230, 230], get_size=14,
            get_alignment_baseline="'bottom'", get_pixel_offset=[0, -14],
        )
        st.pydeck_chart(pdk.Deck(
            layers=[scatter_layer, text_layer], initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/dark-v10",
            tooltip={"text": "{label}"},
        ))

st.divider()
with st.expander("About this model"):
    st.markdown(f"""
    Trained on a Kaggle-sourced synthetic food delivery dataset. The original
    `delivery_duration_minutes` column had no relationship to any other field
    (verified R² ≈ 0), so it was regenerated with a realistic formula — prep
    time (by item) + travel time (distance ÷ effective speed, which depends
    on vehicle and traffic) + dispatch delay (if the driver was offline) +
    noise — before comparing Linear Regression, Random Forest, and Gradient
    Boosting with 5-fold cross-validation. **{model_name}** won and is what's
    serving predictions here. We used HistGradientBoostingRegressor here instead
    of normal boosting because it's much faster on large datasets like this one.
    """)
