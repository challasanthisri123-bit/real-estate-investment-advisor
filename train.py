
# train.py - Essential training script
# Usage: python train.py
import pandas as pd, joblib, os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from math import sqrt

DATA_FILE = "india_housing_prices.csv"
if not os.path.exists(DATA_FILE):
    DATA_FILE = "india_housing_prices_sample.csv"

df = pd.read_csv(DATA_FILE)
df = df.copy()

# Fill missing numeric values
numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

cat_cols = ["City","Property_Type","Furnished_Status","Security","Owner_Type","Availability_Status","Facing"]
num_cols = ["BHK","Size_in_SqFt","Price_per_SqFt","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Public_Transport_Accessibility","Parking_Space"]

X = df[cat_cols + num_cols].fillna(0)
y_clf = df["Good_Investment"]
y_reg = df["Future_Price_5Y"] if "Future_Price_5Y" in df.columns else (df["Price_in_Lakhs"] * 1.4)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=50, random_state=42))
])

reg = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=50, random_state=42))
])

# Train-test split
X_train, X_test, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Train models
clf.fit(X_train, y_train_clf)
pred = clf.predict(X_test)
print("Classifier accuracy:", accuracy_score(y_test_clf, pred))

reg.fit(X_train, y_train_reg)
pred_r = reg.predict(X_test)
print("Regressor RMSE:", sqrt(mean_squared_error(y_test_reg, pred_r)))

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/good_investment_clf.joblib")
joblib.dump(reg, "models/future_price_reg.joblib")
print("Saved models to models/")

