import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
cars = pd.read_csv("car_data.csv")

# Encode categorical variables
le_fuel = LabelEncoder()
cars['Fuel_Type'] = le_fuel.fit_transform(cars['Fuel_Type'])

le_trans = LabelEncoder()
cars['Transmission'] = le_trans.fit_transform(cars['Transmission'])

le_brand = LabelEncoder()
cars['Brand'] = le_brand.fit_transform(cars['Brand'])

X = cars[['Brand', 'Year', 'Mileage', 'Fuel_Type', 'Transmission']]
y = cars['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "car_price_model.pkl")
joblib.dump(le_brand, "le_brand.pkl")
joblib.dump(le_fuel, "le_fuel.pkl")
joblib.dump(le_trans, "le_trans.pkl")
print("✅ Model and encoders saved.")