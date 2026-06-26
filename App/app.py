import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="E-Commerce Purchase Prediction",
    layout="wide",
    page_icon="🛍️",
)

# ── Load model ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/model_prediksi_pembelian_ecommerce.pkl")

model = load_model()

# ── Header ──────────────────────────────────────────────────────────
st.title("🛍️ Will This Visitor Buy?")
st.markdown(
    "Answer questions about a website visitor's session. "
    "The model predicts purchase likelihood."
)

st.divider()

# ── Mode selector ────────────────────────────────────────────────────
mode = st.segmented_control(
    "Input mode",
    ["🟢 Simple"],
    default="🟢 Simple",
    help="Simple: 8 questions. Advanced: all 17 features.",
)

# ── Simple mode ─────────────────────────────────────────────────────
if mode == "🟢 Simple":

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📅 Session info")
        month   = st.selectbox("Month", [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ], index=8)
        weekend = st.toggle("Weekend", value=False)

        st.subheader("👤 Visitor type")
        v_type = st.selectbox("Visitor type", [
            "Returning_Visitor",
            "New_Visitor",
            "Other",
        ])

    with col_b:
        st.subheader("📊 Visitor behavior")
        st.caption("Approximate values are fine.")

        total_pages = st.slider(
            "Product pages visited", 0, 100, 10,
            help="How many product pages did the visitor view?",
        )

        time_on_site = st.slider(
            "Time on product pages (minutes)", 0, 120, 5,
            help="Approximate time spent on product pages.",
        )

        engagement = st.select_slider(
            "Overall engagement",
            options=["Very low", "Low", "Medium", "High", "Very high"],
            value="Medium",
            help="High = browsed many products, added to cart, compared items. Low = viewed 1–2 pages and left.",
        )

        left_quickly = st.toggle(
            "Left after 1–2 pages (bounced)",
            value=False,
        )

        special = st.toggle(
            "Visit near a holiday / special day",
            value=False,
        )

    # ── Page Value (most important predictor) ────────────────────────
    st.subheader("📈 Page Value ⭐ (Most Important)")
    st.info(
        "**This is the most important input.** "
        "In Google Analytics: go to *Behavior → Site Content → All Pages* and look for the "
        "'Page Value' column. It shows the average monetary value of pages visited by users "
        "who later purchased. Pages with high Page Value are the ones that lead to sales."
    )

    page_val = st.slider(
        "Average page value ($)",
        min_value=0.0, max_value=200.0, value=0.0, step=1.0,
        help=(
            "0 = no data / pages don't lead to purchases. "
            "1–20 = moderate (some purchase history). "
            "20+ = high-value pages (frequently leads to purchases). "
            "Check Google Analytics > All Pages > Page Value column."
        ),
    )

    if page_val == 0:
        st.warning(
            "⚠️ Page Value is 0 — predictions will be less accurate. "
            "Try to find this in your analytics tool if possible."
        )

    # ── Build feature vector from simple inputs ──────────────────────
    prod     = total_pages
    prod_dur = time_on_site * 60.0

    engagement_map = {
        "Very low":    (0, 0),
        "Low":         (1, 0),
        "Medium":      (2, 1),
        "High":        (3, 2),
        "Very high":   (5, 3),
    }
    admin, info = engagement_map[engagement]
    admin_dur = admin * 30.0
    info_dur  = info  * 30.0

    bounce = 0.15 if left_quickly else 0.005
    exit_r = 0.15 if left_quickly else 0.02
    special_val = 0.8 if special else 0.0

    # Technical defaults (most common values)
    os_id   = 2
    browser = 2
    region  = 1
    traffic = 2

    input_data = pd.DataFrame({
        "Administrative":          [admin],
        "Administrative_Duration": [admin_dur],
        "Informational":           [info],
        "Informational_Duration":  [info_dur],
        "ProductRelated":          [prod],
        "ProductRelated_Duration": [prod_dur],
        "BounceRates":             [bounce],
        "ExitRates":              [exit_r],
        "PageValues":              [page_val],
        "SpecialDay":             [special_val],
        "Month":                  [month],
        "OperatingSystems":        [os_id],
        "Browser":                [browser],
        "Region":                 [region],
        "TrafficType":            [traffic],
        "VisitorType":            [v_type],
        "Weekend":                [weekend],
    })

# ── Advanced mode ────────────────────────────────────────────────────
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Page Visits")
        c1, c2 = st.columns(2)
        with c1:
            admin    = st.number_input("Admin pages", min_value=0, value=0)
            info     = st.number_input("Info pages", min_value=0, value=0)
            prod     = st.number_input("Product pages", min_value=0, value=0)
        with c2:
            admin_dur = st.number_input("Admin duration (s)", min_value=0.0, value=0.0, format="%.1f")
            info_dur  = st.number_input("Info duration (s)", min_value=0.0, value=0.0, format="%.1f")
            prod_dur  = st.number_input("Product duration (s)", min_value=0.0, value=0.0, format="%.1f")

        st.subheader("📈 Behavior")
        bounce   = st.slider("Bounce rate", 0.0, 1.0, 0.02, 0.01)
        exit_r   = st.slider("Exit rate", 0.0, 1.0, 0.02, 0.01)
        page_val = st.number_input("Page value", min_value=0.0, value=0.0, format="%.2f")
        special  = st.slider("Special day proximity", 0.0, 1.0, 0.0, 0.1)

    with col2:
        st.subheader("🗓️ Session Context")
        month   = st.selectbox("Month", [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ])
        weekend = st.toggle("Weekend", value=False)

        st.subheader("👤 Visitor Type")
        v_type = st.selectbox("Visitor type", [
            "Returning_Visitor", "New_Visitor", "Other",
        ])

        st.subheader("🔧 Technical (anonymized IDs)")
        st.caption("Defaults are the most common values. Change only if you know the actual ID.")
        os_id   = st.selectbox("OS", list(range(1, 9)))
        browser = st.selectbox("Browser", list(range(1, 14)))
        region  = st.selectbox("Region", list(range(1, 10)))
        traffic = st.selectbox("Traffic type", list(range(1, 21)))

    input_data = pd.DataFrame({
        "Administrative":          [admin],
        "Administrative_Duration": [admin_dur],
        "Informational":           [info],
        "Informational_Duration":  [info_dur],
        "ProductRelated":          [prod],
        "ProductRelated_Duration": [prod_dur],
        "BounceRates":             [bounce],
        "ExitRates":              [exit_r],
        "PageValues":              [page_val],
        "SpecialDay":              [special],
        "Month":                   [month],
        "OperatingSystems":        [os_id],
        "Browser":                [browser],
        "Region":                 [region],
        "TrafficType":            [traffic],
        "VisitorType":            [v_type],
        "Weekend":                [weekend],
    })

# ── Predict ─────────────────────────────────────────────────────────
st.divider()

_, center, _ = st.columns([1, 2, 1])
with center:
    predict_btn = st.button("🚀  Predict Purchase", width='stretch', type="primary")

if predict_btn:
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### Result")
    if prediction == 1:
        st.success("**Likely to purchase**")
    else:
        st.warning("**Unlikely to purchase**")

    st.metric("Purchase probability", f"{probability:.1%}")
    st.progress(float(probability), text=f"Confidence: {probability:.0%} purchase likelihood")

    with st.expander("📋 What was sent to the model"):
        display_df = input_data.T.rename(columns={0: "Value"})
        display_df.index.name = "Feature"
        st.dataframe(display_df.astype(str), width='stretch')

st.markdown("---")
st.caption(
    "Model: Gradient Boosting · trained on UCI Online Shoppers Intention dataset."
)
