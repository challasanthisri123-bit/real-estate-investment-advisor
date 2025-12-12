
# train.py - Essential training script
# Usage: python train.py
import pandas as pd, joblib, os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

DATA_FILE = "india_housing_prices.csv"
if not os.path.exists(DATA_FILE):
    DATA_FILE = "india_housing_prices_sample.csv"  # fallback sample provided in the package

df = pd.read_csv(DATA_FILE)
# Minimal preprocessing: select numeric features and simple encodings
df = df.copy()
# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Quick categorical encoding for small feature set
cat_cols = ["City","Property_Type","Furnished_Status","Security","Owner_Type","Availability_Status","Facing"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype(str).fillna("NA")
        df[c] = df[c].factorize()[0]

# Features for models -- keep it small & essential
features = ["BHK","Size_in_SqFt","Price_per_SqFt","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Public_Transport_Accessibility","Parking_Space"]
for c in ["City","Property_Type","Furnished_Status","Security","Owner_Type","Availability_Status","Facing"]:
    if c in df.columns:
        features.append(c)

X = df[features].fillna(0)
y_clf = df["Good_Investment"]
y_reg = df["Future_Price_5Y"] if "Future_Price_5Y" in df.columns else (df["Price_in_Lakhs"] * 1.4)

# Train-test split
X_train, X_test, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Train models
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train_clf)
pred = clf.predict(X_test)
print("Classifier accuracy:", accuracy_score(y_test_clf, pred))

reg = RandomForestRegressor(n_estimators=50, random_state=42)
reg.fit(X_train, y_train_reg)
pred_r = reg.predict(X_test)   
from math import sqrt
print("Regressor RMSE:", sqrt(mean_squared_error(y_test_reg, pred_r)))

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/good_investment_clf.joblib")
joblib.dump(reg, "models/future_price_reg.joblib")
print("Saved models to models/")
