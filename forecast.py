import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your data
df = pd.read_csv("energy_usage_v2.csv", parse_dates=["Date"])
print(df.head())

# Create "Next_Day_Usage" column
df["Next_Day_Usage"] = df["Energy_Usage_kWh"].shift(-1)
df = df.dropna()

# Prepare data
X = df[["Energy_Usage_kWh"]]
y = df["Next_Day_Usage"]

# Train/test split
train_size = int(len(X) * 0.8)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict tomorrow
latest = df["Energy_Usage_kWh"].iloc[-1]
pred = model.predict([[latest]])
print(f"✅ Tomorrow's predicted usage: {pred[0]:.2f} kWh")

# Forecast for 7 days ahead
days_to_predict = 7
predictions = []

# Start from the most recent usage value
current_usage = df["Energy_Usage_kWh"].iloc[-1]

for i in range(days_to_predict):
    next_pred = model.predict([[current_usage]])[0]
    predictions.append(round(next_pred, 2))
    current_usage = next_pred  # use the prediction as next input

# Show the results
print("\n📅 7-Day Energy Usage Forecast:")
for i, value in enumerate(predictions, 1):
    print(f"Day {i}: {value:.2f} kWh")

import matplotlib.pyplot as plt

plt.plot(range(1, 8), predictions, marker='o', linestyle='--')
plt.title("7-Day Energy Usage Forecast")
plt.xlabel("Day")
plt.ylabel("Predicted Usage (kWh)")
plt.grid(True)
plt.show()
