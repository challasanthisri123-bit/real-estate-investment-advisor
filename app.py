
# Streamlit app (app.py)
# Run: streamlit run app.py
import streamlit as st
import pandas as pd, joblib, os, numpy as np

st.set_page_config(page_title="Real Estate Investment Advisor", layout="centered")

st.title("Real Estate Investment Advisor - Quick Demo")
st.markdown("Enter property details to get: 1) Is it a Good Investment? 2) Estimated price after 5 years.")

# Load models if available
clf_path = "models/good_investment_clf.joblib"
reg_path = "models/future_price_reg.joblib"
clf = joblib.load(clf_path) if os.path.exists(clf_path) else None
reg = joblib.load(reg_path) if os.path.exists(reg_path) else None

with st.form("property_form"):
    city = st.selectbox("City", ["Mumbai","Bengaluru","Delhi","Hyderabad","Chennai"])
    prop_type = st.selectbox("Property Type", ["Apartment","Villa","House"])
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    sqft = st.number_input("Size (SqFt)", min_value=100, max_value=10000, value=900)
    price_lakhs = st.number_input("Current Price (in Lakhs)", min_value=1.0, value=50.0, step=0.5)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2015)
    schools = st.number_input("Nearby Schools (count)", min_value=0, max_value=50, value=2)
    hospitals = st.number_input("Nearby Hospitals (count)", min_value=0, max_value=50, value=1)
    public_transport = st.selectbox("Public Transport Access (0=Low,1=Med,2=High)", [0,1,2], index=1)
    parking = st.selectbox("Parking spaces", [0,1,2], index=1)
    security = st.selectbox("Security", ["Gated","CCTV","Guard","None"])
    furnished = st.selectbox("Furnished Status", ["Unfurnished","Semi","Fully"])
    availability = st.selectbox("Availability", ["Available","Under Construction","Sold"])
    facing = st.selectbox("Facing", ["North","South","East","West"])
    submitted = st.form_submit_button("Predict")
    
if submitted:
    price_per_sqft = (price_lakhs * 100000) / max(1, sqft)
    age = 2025 - year_built
    # Minimal encoding consistent with train.py's factorize order - we use simple hashing via dicts for demo
    enc = {}
    def encode(col, val):
        # deterministic simple encoding based on hashing mapping in app
        return abs(hash(col+str(val))) % 20  # small bucket encoding for demo
    
    feat = {
        "BHK": bhk,
        "Size_in_SqFt": sqft,
        "Price_per_SqFt": price_per_sqft,
        "Age_of_Property": age,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": public_transport,
        "Parking_Space": parking,
        "City": encode("City", city),
        "Property_Type": encode("Property_Type", prop_type),
        "Furnished_Status": encode("Furnished_Status", furnished),
        "Security": encode("Security", security),
        "Owner_Type": encode("Owner_Type", "Individual"),
        "Availability_Status": encode("Availability_Status", availability),
        "Facing": encode("Facing", facing)
    }
    X = pd.DataFrame([feat.values()], columns=list(feat.keys()))
    st.write("Input features:", X.T)
    if clf is not None and reg is not None:
        pred_clf = clf.predict(X)[0]
        pred_reg = reg.predict(X)[0]
        st.success(f"Good Investment? -> {'Yes' if pred_clf==1 else 'No'}")
        st.info(f"Estimated Price after 5 years (Lakhs): {round(pred_reg,2)}")
    else:
        st.warning("Models not found. Please run `python train.py` first to create `models/` directory.")
