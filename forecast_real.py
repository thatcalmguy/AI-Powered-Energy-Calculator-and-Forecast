import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load your data
df = pd.read_csv("energy_usage_v2.csv", parse_dates=["Date"])

# STEP 1: Feature Engineering
df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday

# Simulate extra features:
np.random.seed(42)  # For reproducibility

df["Weather"] = np.random.choice(["Hot", "Mild", "Cold"], size=len(df))
df["Family_Presence"] = np.random.randint(1, 6, size=len(df))       # 1–5 people
df["Appliance_Usage"] = np.random.randint(1, 8, size=len(df))       # 1–7 appliances

# Encode weather (one-hot)
df = pd.get_dummies(df, columns=["Weather"])

# STEP 2: Create label
df["Next_Day_Usage"] = df["Energy_Usage_kWh"].shift(-1)
df = df.dropna()

# STEP 3: Prepare features (X) and label (y)
feature_cols = ["Energy_Usage_kWh", "DayOfWeek", "Family_Presence", "Appliance_Usage",
                "Weather_Cold", "Weather_Hot", "Weather_Mild"]
X = df[feature_cols]
y = df["Next_Day_Usage"]

# STEP 4: Train/test split
train_size = int(len(X) * 0.8)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# STEP 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 6: Predict next 7 days (recursive)
print("\n📅 7-Day Forecast with Real-Life Factors:")

# Start with last known row
last_row = df.iloc[-1].copy()
predictions = []

for i in range(7):
    row_features = last_row[feature_cols].values.reshape(1, -1)
    next_pred = model.predict(row_features)[0]
    predictions.append(round(next_pred, 2))

    # Simulate next day's features
    last_row["Energy_Usage_kWh"] = next_pred
    last_row["DayOfWeek"] = (last_row["DayOfWeek"] + 1) % 7
    last_row["Family_Presence"] = np.random.randint(1, 6)
    last_row["Appliance_Usage"] = np.random.randint(1, 8)
    weather = np.random.choice(["Cold", "Hot", "Mild"])
    last_row["Weather_Cold"] = int(weather == "Cold")
    last_row["Weather_Hot"] = int(weather == "Hot")
    last_row["Weather_Mild"] = int(weather == "Mild")

# Display forecast
for i, val in enumerate(predictions, 1):
    print(f"Day {i}: {val:.2f} kWh")

# Optional: Plot
plt.plot(range(1, 8), predictions, marker='o', linestyle='--')
plt.title("7-Day Energy Usage Forecast (Enhanced)")
plt.xlabel("Day")
plt.ylabel("Predicted Usage (kWh)")
plt.grid(True)
plt.show()