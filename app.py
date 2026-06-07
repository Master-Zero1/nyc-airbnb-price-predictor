import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Red Hat Display', sans-serif;
    background-color: #0a0a0f;
    color: #ffffff;
}

/* hide streamlit default header */
#MainMenu, header, footer {visibility: hidden;}

.block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* hero header */
.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0a2e 50%, #0a0a0f 100%);
    border-bottom: 3px solid #e63946;
    padding: 2.5rem 0 2rem 0;
    margin-bottom: 2.5rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}
.hero h1 span { color: #e63946; }
.hero p {
    color: #aaaaaa;
    font-size: 1.05rem;
    margin-top: 0.4rem;
}
.badge {
    display: inline-block;
    background: #e63946;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    letter-spacing: 0.5px;
}

/* cards */
.card {
    background: #13131f;
    border: 1px solid #2a2a3f;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-red {
    background: #13131f;
    border: 1px solid #e63946;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card h3 {
    color: #e63946;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.price-display {
    font-size: 3.8rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
}
.price-range {
    color: #aaaaaa;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}
.price-unit {
    font-size: 1.2rem;
    color: #e63946;
    font-weight: 700;
}

/* section labels */
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #e63946;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2a3f;
}

/* predict button */
div.stButton > button {
    background: #e63946;
    color: white;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    width: 100%;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: #c1121f;
    transform: translateY(-1px);
}

/* input styling */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div {
    background: #1e1e2f;
    border-radius: 8px;
}

/* neighbour comparison bars */
.bar-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 10px;
}
.bar-label { width: 90px; font-size: 0.85rem; color: #cccccc; }
.bar-track {
    flex: 1;
    background: #2a2a3f;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}
.bar-fill {
    height: 10px;
    border-radius: 4px;
    background: #e63946;
}
.bar-value { width: 70px; font-size: 0.85rem; color: #ffffff; text-align: right; }

/* metric boxes */
.metric-box {
    background: #1e1e2f;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-box .val {
    font-size: 1.4rem;
    font-weight: 900;
    color: #e63946;
}
.metric-box .lbl {
    font-size: 0.7rem;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="stTabs"] button {
    color: #aaaaaa;
    font-weight: 600;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e63946;
    border-bottom: 2px solid #e63946;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model       = joblib.load('models/xgboost_tuned_final.pkl')
    scaler      = joblib.load('models/scaler.pkl')
    one_enc     = joblib.load('models/one_enc.pkl')
    ord_enc     = joblib.load('models/ord_enc.pkl')
    neigh_means = joblib.load('models/neigh_means.pkl')
    feat_cols   = joblib.load('models/feature_columns.pkl')
    return model, scaler, one_enc, ord_enc, neigh_means, feat_cols

model, scaler, one_enc, ord_enc, neigh_means, feat_cols = load_models()

if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'price' not in st.session_state:
    st.session_state.price = 0
if 'low' not in st.session_state:
    st.session_state.low = 0
if 'high' not in st.session_state:
    st.session_state.high = 0
if 'df_input' not in st.session_state:
    st.session_state.df_input = None
if 'params' not in st.session_state:
    st.session_state.params = None

BOROUGHS = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
ROOM_TYPES = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
PROPERTY_TYPES = [
    'Entire rental unit', 'Private room in rental unit',
    'Entire condo', 'Private room in home', 'Entire home',
    'Room in hotel', 'Private room in condo', 'Entire serviced apartment',
    'Room in boutique hotel', 'Private room in hostel'
]
AMENITIES = ['has_wifi', 'has_kitchen', 'has_washer', 'has_dryer',
             'has_ac', 'has_heating', 'has_tv', 'has_elevator',
             'has_dishwasher', 'has_gym', 'has_pool', 'has_parking']
AMENITY_LABELS = ['WiFi', 'Kitchen', 'Washer', 'Dryer',
                  'AC', 'Heating', 'TV', 'Elevator',
                  'Dishwasher', 'Gym', 'Pool', 'Parking']

BOROUGH_COORDS = {
    'Manhattan': (-73.971, 40.776),
    'Brooklyn':  (-73.944, 40.678),
    'Queens':    (-73.867, 40.728),
    'Bronx':     (-73.865, 40.837),
    'Staten Island': (-74.151, 40.579)
}

SCALE_COLS = ['hosts_time_as_host_years', 'accommodates', 'bathrooms', 'bedrooms',
              'maximum_nights', 'maximum_nights_avg_ntm', 'number_of_reviews',
              'estimated_occupancy_l365d', 'days_since_first_review',
              'days_since_last_review', 'calculated_host_listings_count',
              'amenity_count', 'availability_rate', 'total_availability_score',
              'review_scores_cleanliness', 'review_scores_location',
              'review_score_avg', 'neighbourhood_cleansed']

def build_input(params):
    row = {col: 0 for col in feat_cols}

    # basic numerics
    row['accommodates']                = params['accommodates']
    row['bedrooms']                    = params['bedrooms']
    row['bathrooms']                   = params['bathrooms']
    row['amenity_count']               = sum(params[a] for a in AMENITIES)
    row['number_of_reviews']           = params['number_of_reviews']
    row['review_scores_cleanliness']   = params['review_scores_cleanliness']
    row['review_scores_location']      = params['review_scores_location']
    row['review_score_avg']            = params['review_score_avg']
    row['host_is_superhost']           = int(params['host_is_superhost'])
    row['has_availability']            = 1
    row['calculated_host_listings_count'] = params['host_listings']
    row['maximum_nights']              = 365
    row['maximum_nights_avg_ntm']      = 365
    row['availability_rate']           = 0.5
    row['total_availability_score']    = 0.5
    row['estimated_occupancy_l365d']   = 180
    row['days_since_first_review']     = 365
    row['days_since_last_review']      = 30
    row['hosts_time_as_host_years']    = params['host_years']

    # engineered features
    row['bathrooms_per_guest'] = params['bathrooms'] / max(params['accommodates'], 1)
    row['bedrooms_per_guest']  = params['bedrooms']  / max(params['accommodates'], 1)
    row['amenity_density']     = row['amenity_count'] / max(params['accommodates'], 1)
    row['listing_age_years']   = 2.0
    row['is_recently_active']  = 1
    row['reviews_per_year']    = max(params['number_of_reviews'] / 2.0, 0.1)

    # amenities
    for a in AMENITIES:
        if a in row:
            row[a] = int(params[a])

    # neighbourhood target encoding
    borough = params['neighbourhood_group_cleansed']
    lon, lat = BOROUGH_COORDS[borough]
    row['longitude'] = lon
    row['latitude']  = lat
    neigh_val = neigh_means.get(borough, np.mean(list(neigh_means.values())))
    row['neighbourhood_cleansed'] = neigh_val

    df = pd.DataFrame([row])

    # ordinal encode host_experience_level
    exp_map = {'New': 0, 'Intermediate': 1, 'Experienced': 2}
    df['host_experience_level'] = exp_map.get(params['host_experience_level'], 2)

    # one-hot encode
    ohe_df = pd.DataFrame(
        one_enc.transform(pd.DataFrame([[params['room_type'],
                                          params['neighbourhood_group_cleansed'],
                                          params['property_type']]],
                           columns=['room_type',
                                    'neighbourhood_group_cleansed',
                                    'property_type'])),
        columns=one_enc.get_feature_names_out(
            ['room_type', 'neighbourhood_group_cleansed', 'property_type'])
    )
    for col in ohe_df.columns:
        if col in df.columns:
            df[col] = ohe_df[col].values

    # scale
    scale_present = [c for c in SCALE_COLS if c in df.columns]
    df[scale_present] = scaler.transform(df[scale_present])

    df = df[feat_cols]
    return df

def predict_price(df):
    log_pred = model.predict(df)[0]
    price    = np.expm1(log_pred)
    low      = np.expm1(log_pred - 0.19)
    high     = np.expm1(log_pred + 0.19)
    return price, low, high

st.markdown("""
<div class="hero">
  <h1>🏙️ NYC Airbnb <span>Price</span> Predictor</h1>
  <p>
    <span class="badge">XGBoost</span>
    <span class="badge">R² = 0.768</span>
    <span class="badge">$188 avg error</span>
    &nbsp; Powered by 20,000+ NYC listings · April 2026 scrape
  </p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown('<div class="section-label">📍 Listing Details</div>',
                unsafe_allow_html=True)

    with st.container():
        borough = st.selectbox("Borough", BOROUGHS)
        room_type = st.selectbox("Room Type", ROOM_TYPES)
        property_type = st.selectbox("Property Type", PROPERTY_TYPES)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">🛏️ Size</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        accommodates = st.number_input("Guests", 1, 16, 2)
    with c2:
        bedrooms = st.number_input("Bedrooms", 0, 10, 1)
    with c3:
        bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">⭐ Reviews</div>',
                unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        review_score_avg        = st.slider("Overall score", 1.0, 5.0, 4.5, 0.1)
        review_scores_cleanliness = st.slider("Cleanliness", 1.0, 5.0, 4.5, 0.1)
    with c5:
        review_scores_location  = st.slider("Location score", 1.0, 5.0, 4.7, 0.1)
        number_of_reviews       = st.number_input("# Reviews", 0, 2000, 50)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">👤 Host</div>',
                unsafe_allow_html=True)
    c6, c7, c8 = st.columns(3)
    with c6:
        host_experience_level = st.selectbox("Experience",
                                              ['New', 'Intermediate', 'Experienced'])
    with c7:
        host_years    = st.number_input("Years hosting", 0, 15, 3)
    with c8:
        host_listings = st.number_input("# Listings", 1, 100, 1)
    host_is_superhost = st.checkbox("⚡ Superhost")

    st.markdown('<div class="section-label" style="margin-top:1.2rem">🛎️ Amenities</div>',
                unsafe_allow_html=True)
    amenity_vals = {}
    cols = st.columns(3)
    for i, (key, label) in enumerate(zip(AMENITIES, AMENITY_LABELS)):
        with cols[i % 3]:
            amenity_vals[key] = st.checkbox(label, value=(key in ['has_wifi',
                                                                    'has_kitchen',
                                                                    'has_heating']))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 PREDICT PRICE")

with right:
    if predict_btn:
        params = {
            'neighbourhood_group_cleansed': borough,
            'room_type':          room_type,
            'property_type':      property_type,
            'accommodates':       accommodates,
            'bedrooms':           bedrooms,
            'bathrooms':          bathrooms,
            'review_score_avg':   review_score_avg,
            'review_scores_cleanliness': review_scores_cleanliness,
            'review_scores_location':    review_scores_location,
            'number_of_reviews':  number_of_reviews,
            'host_experience_level': host_experience_level,
            'host_years':         host_years,
            'host_listings':      host_listings,
            'host_is_superhost':  host_is_superhost,
            **amenity_vals
        }
        with st.spinner("Running model..."):
            df_input         = build_input(params)
            price, low, high = predict_price(df_input)
            st.session_state.predicted = True
            st.session_state.price     = price
            st.session_state.low       = low
            st.session_state.high      = high
            st.session_state.df_input  = df_input
            st.session_state.params    = params

    if st.session_state.predicted:
        price    = st.session_state.price
        low      = st.session_state.low
        high     = st.session_state.high
        df_input = st.session_state.df_input
        params   = st.session_state.params

        with st.spinner("Running model..."):
            df_input         = build_input(params)
            price, low, high = predict_price(df_input)
            # save to session state
            st.session_state.predicted = True
            st.session_state.price     = price
            st.session_state.low       = low
            st.session_state.high      = high
            st.session_state.df_input  = df_input
            st.session_state.params    = params

        st.markdown(f"""
        <div class="card-red">
          <h3>💰 Predicted Price</h3>
          <div class="price-display">${price:,.0f}
            <span class="price-unit">/night</span>
          </div>
          <div class="price-range">
            Confidence range: ${low:,.0f} – ${high:,.0f} per night
          </div>
        </div>
        """, unsafe_allow_html=True)

        monthly = price * 20
        yearly  = price * 200
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="val">${monthly:,.0f}</div><div class="lbl">Est. Monthly</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="val">${yearly:,.0f}</div><div class="lbl">Est. Yearly</div></div>', unsafe_allow_html=True)
        with m3:
            pct = int((price / 175) * 100)
            st.markdown(f'<div class="metric-box"><div class="val">{pct}%</div><div class="lbl">vs NYC Avg ($175)</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🔍 SHAP Breakdown",
                                     "🗺️ Borough Compare",
                                     "⚡ What-If Simulator"])

        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Why did the model predict this price?**")
            with st.spinner("Computing SHAP..."):
                explainer   = shap.TreeExplainer(model)
                shap_vals   = explainer.shap_values(df_input)
                shap_series = pd.Series(shap_vals[0],
                                        index=feat_cols).abs().sort_values(ascending=False).head(10)
                shap_raw    = pd.Series(shap_vals[0], index=feat_cols)

            fig, ax = plt.subplots(figsize=(7, 4),
                                   facecolor='#13131f')
            ax.set_facecolor('#13131f')
            top     = shap_raw.abs().sort_values(ascending=False).head(10)
            top_raw = shap_raw[top.index]
            colors  = ['#e63946' if v > 0 else '#4a90d9' for v in top_raw.values]
            ax.barh(range(len(top)), top_raw.values[::-1],
                    color=colors[::-1], edgecolor='none', height=0.6)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index[::-1], color='white', fontsize=9)
            ax.set_xlabel('SHAP value (impact on log price)', color='#aaaaaa', fontsize=9)
            ax.tick_params(colors='#aaaaaa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#2a2a3f')
            ax.axvline(0, color='#2a2a3f', linewidth=1)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("""
            <small style='color:#888'>
            🔴 Red = pushes price <b>up</b> &nbsp;|&nbsp;
            🔵 Blue = pushes price <b>down</b>
            </small>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Same listing across all 5 NYC boroughs:**")
            borough_prices = {}
            for b in BOROUGHS:
                p2             = params.copy()
                p2['neighbourhood_group_cleansed'] = b
                df2            = build_input(p2)
                pr, _, _       = predict_price(df2)
                borough_prices[b] = pr

            max_p = max(borough_prices.values())
            bars_html = ""
            for b, p in sorted(borough_prices.items(),
                                key=lambda x: x[1], reverse=True):
                width  = int((p / max_p) * 100)
                border = "border: 1px solid #e63946;" if b == borough else ""
                bars_html += f"""
                <div class="bar-row">
                  <div class="bar-label">{b}</div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:{width}%;{border}"></div>
                  </div>
                  <div class="bar-value">${p:,.0f}</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown(
                f"<small style='color:#888'>Your borough ({borough}) is highlighted in red.</small>",
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Toggle features and see how price changes instantly:**")

            w1, w2 = st.columns(2)
            with w1:
                wi_superhost = st.checkbox("⚡ Superhost status",
                                           value=host_is_superhost,
                                           key="wi_sh")
                wi_gym       = st.checkbox("🏋️ Add Gym",
                                           value=amenity_vals.get('has_gym', False),
                                           key="wi_gym")
                wi_pool      = st.checkbox("🏊 Add Pool",
                                           value=amenity_vals.get('has_pool', False),
                                           key="wi_pool")
            with w2:
                wi_elevator  = st.checkbox("🛗 Add Elevator",
                                           value=amenity_vals.get('has_elevator', False),
                                           key="wi_el")
                wi_dishwasher= st.checkbox("🍽️ Add Dishwasher",
                                           value=amenity_vals.get('has_dishwasher', False),
                                           key="wi_dw")
                wi_parking   = st.checkbox("🚗 Add Parking",
                                           value=amenity_vals.get('has_parking', False),
                                           key="wi_pk")

            wi_params = params.copy()
            wi_params.update({
                'host_is_superhost': wi_superhost,
                'has_gym':       wi_gym,
                'has_pool':      wi_pool,
                'has_elevator':  wi_elevator,
                'has_dishwasher':wi_dishwasher,
                'has_parking':   wi_parking,
            })
            wi_df          = build_input(wi_params)
            wi_price, _, _ = predict_price(wi_df)
            delta          = wi_price - price
            delta_str      = f"+${delta:,.0f}" if delta >= 0 else f"-${abs(delta):,.0f}"
            color          = "#e63946" if delta >= 0 else "#4a90d9"

            st.markdown(f"""
            <div style="margin-top:1rem; padding:1rem;
                        background:#1e1e2f; border-radius:8px; text-align:center;">
              <div style="font-size:0.8rem; color:#888; text-transform:uppercase;
                          letter-spacing:1px;">What-If Price</div>
              <div style="font-size:2.5rem; font-weight:900;
                          color:#ffffff;">${wi_price:,.0f}
                <span style="font-size:1rem; color:#888">/night</span>
              </div>
              <div style="font-size:1.1rem; font-weight:700;
                          color:{color};">{delta_str} vs current</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif not st.session_state.predicted:
        st.markdown("""
        <div class="card" style="text-align:center; padding:4rem 2rem;">
          <div style="font-size:4rem;">🏙️</div>
          <div style="font-size:1.3rem; font-weight:700;
                      color:#ffffff; margin-top:1rem;">
            Fill in your listing details
          </div>
          <div style="color:#888; margin-top:0.5rem;">
            Set the inputs on the left and click<br>
            <span style="color:#e63946; font-weight:700">PREDICT PRICE</span>
            to see results
          </div>
        </div>
        """, unsafe_allow_html=True)