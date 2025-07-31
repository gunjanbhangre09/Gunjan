import pandas as pd from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression from sklearn.preprocessing import OneHotEncoder from sklearn.compose import ColumnTransformer from sklearn.pipeline import Pipeline from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("drip_irrigation_projects_100.csv")

target = "Estimated Cost (INR)" features = [ "Location", "Crop Type", "Area Covered (acres)", "Water Source", "Pump Capacity (HP)", "Pipe Material", "Pipe Diameter (mm)", "Estimated Water Usage (litres/day)" ]

X = df[features] y = df[target]

X.loc[:, "Pump Capacity (HP)"] = pd.to_numeric(X["Pump Capacity (HP)"], errors='coerce')

categorical_features = ["Location", "Crop Type", "Water Source", "Pipe Material"] numerical_features = ["Area Covered (acres)", "Pump Capacity (HP)", "Pipe Diameter (mm)", "Estimated Water Usage (litres/day)"]

preprocessor = ColumnTransformer( transformers=[ ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features) ], remainder='passthrough' )

model = Pipeline(steps=[ ("preprocessor", preprocessor), ("regressor", LinearRegression()) ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse) print("R² Score:", r2)
