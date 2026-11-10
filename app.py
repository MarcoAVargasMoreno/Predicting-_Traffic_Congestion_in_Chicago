# ========================================
# Chicago Traffic Congestion Prediction App
# MIT 105 - Assessment 2
# ========================================

import streamlit as st
import pandas as pd
import pydeck as pdk
from pycaret.classification import load_model, predict_model

# Page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Chicago Traffic Congestion",
    page_icon="🚦",
    layout="centered"
)

# Cache the model so it loads only once (not on every rerun)
@st.cache_resource
def load_catboost_model():
    """Load the trained CatBoost pipeline from disk."""
    model = load_model("catboost_congestion_model")
    return model

model = load_catboost_model()

# Cache the scaler so it loads only once (same as the model)
@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler used during training.
    This scaler must be applied to the raw user inputs before prediction,
    to avoid training-serving skew (the model was trained on scaled features).
    """
    import joblib
    scaler = joblib.load("scaler.pkl")
    return scaler

scaler = load_scaler()

# Columns that were scaled during training (must match the notebook exactly)
COLS_TO_SCALE = ["month", "day_of_week", "hour", "length"]

# Cache the segments metadata (loaded once, reused)
@st.cache_data
def load_segments_metadata():
    """Load the segments metadata parquet and build readable labels."""
    df = pd.read_parquet("inputs/chicago_segments_metadata.parquet")
    # Build a readable label for each segment
    df["label"] = (
        df["street"] + " — "
        + df["from_street"] + " → "
        + df["to_street"]
        + " (" + df["direction"] + ")"
    )
    # Sort by street name so the dropdown is easier to browse
    df = df.sort_values("label").reset_index(drop=True)
    return df

segments_df = load_segments_metadata()


# Title and intro
st.title("🚦 Chicago Traffic Congestion Prediction")
st.markdown(
    "Predict traffic congestion level on Chicago street segments "
    "using a machine learning model trained on data from the official Chicago data portal."
)

st.divider()

# Inputs section
st.subheader("Select your trip details")

# Segment dropdown
selected_label = st.selectbox(
    "Street segment",
    options=segments_df["label"].tolist(),
    help="Choose the street segment you plan to travel on."
)

# Get the full row of the selected segment
selected_segment = segments_df[segments_df["label"] == selected_label].iloc[0]

# Date and hour inputs (two columns side by side)
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Travel date",
        help="Choose the date you plan to travel. We use it to extract the day of the week and the month."
    )

with col2:
    selected_hour = st.selectbox(
        "Hour of the day",
        options=list(range(24)),
        index=8,  # default to 8 AM
        format_func=lambda x: f"{x:02d}:00",
        help="Choose the hour of your trip (24-hour format)."
    )

# ========================================
# Build the feature row for the model
# ========================================

# Extract month and day_of_week from the date
month = selected_date.month

# IMPORTANT: Convert Python weekday (Monday=0) to Socrata (Sunday=0)
# The model was trained with Socrata convention: Sunday=0, Monday=1, ..., Saturday=6
python_weekday = selected_date.weekday()  # Monday=0, ..., Sunday=6
day_of_week = (python_weekday + 1) % 7    # now Sunday=0, Monday=1, ..., Saturday=6

# Derive binary flags
is_weekend = 1 if day_of_week in [0, 6] else 0
is_rush_hour = 1 if selected_hour in [7, 8, 9, 16, 17, 18] else 0

# All possible values for the one-hot encoded columns
# (must match exactly what the model was trained with)
ALL_DIRECTIONS = ["EB", "NB", "NE", "NW", "SB", "SE", "SW", "WB"]
ALL_HEADINGS = ["E", "N", "S", "W"]

# Build the base row with numeric and binary features
row_data = {
    "month": month,
    "day_of_week": day_of_week,
    "hour": selected_hour,
    "is_weekend": is_weekend,
    "is_rush_hour": is_rush_hour,
    "length": selected_segment["length"],
}

# One-hot encode direction (set 1 for the selected, 0 for the rest)
for d in ALL_DIRECTIONS:
    row_data[f"direction_{d}"] = 1 if selected_segment["direction"] == d else 0

# One-hot encode street_heading (same idea)
for h in ALL_HEADINGS:
    row_data[f"street_heading_{h}"] = 1 if selected_segment["street_heading"] == h else 0

# Build the final input DataFrame (one row, 18 columns)
input_row = pd.DataFrame([row_data])

st.divider()

# Temporary debug: show input row before prediction
with st.expander("🔍 Debug: input to model"):
    st.markdown("**Raw input (before scaling):**")
    st.dataframe(input_row)
    st.markdown("**Scaled input (what the model actually receives):**")
    input_row_scaled_debug = input_row.copy()
    input_row_scaled_debug[COLS_TO_SCALE] = scaler.transform(input_row[COLS_TO_SCALE])
    st.dataframe(input_row_scaled_debug)


# Predict button
if st.button("🔮 Predict congestion", type="primary", use_container_width=True):
    # Apply the same scaling that was used during training.
    # Only the 4 numeric columns are scaled; binary flags and one-hot stay as 0/1.
    input_row_scaled = input_row.copy()
    input_row_scaled[COLS_TO_SCALE] = scaler.transform(input_row[COLS_TO_SCALE])

    # Run the model on the scaled input
    prediction = predict_model(model, data=input_row_scaled)    
    predicted_class = prediction["prediction_label"].iloc[0]
    predicted_score = prediction["prediction_score"].iloc[0]

    # Show result with color based on the class
    if predicted_class == "Heavy":
        st.error(f"🔴 Predicted congestion: **{predicted_class}**")
    elif predicted_class == "Moderate":
        st.warning(f"🟡 Predicted congestion: **{predicted_class}**")
    else:  # Free Flow
        st.success(f"🟢 Predicted congestion: **{predicted_class}**")

    st.caption(f"Model confidence: {predicted_score:.1%}")
    
    # Mini-map showing the selected segment location, colored by predicted class
    st.markdown("### Segment location")
    mid_lat = (float(selected_segment["start_latitude"]) + float(selected_segment["end_latitude"])) / 2
    mid_lon = (float(selected_segment["start_longitude"]) + float(selected_segment["end_longitude"])) / 2
    
    # Color of the marker depends on the predicted class
    if predicted_class == "Heavy":
        marker_color = [220, 50, 50]       # red
    elif predicted_class == "Moderate":
        marker_color = [240, 180, 40]      # amber
    else:
        marker_color = [60, 180, 75]       # green
        
    map_data = pd.DataFrame({"lat": [mid_lat], "lon": [mid_lon]})
    
    layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position=["lon", "lat"],
    get_fill_color=marker_color,
    get_radius=120,
    pickable=False,
    )
    
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=13)
    
    st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v10",
    #map_style="mapbox://styles/mapbox/light-v9",
    ))