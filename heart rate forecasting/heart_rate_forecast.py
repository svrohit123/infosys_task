# TASK 1: Heart Rate Forecasting
# Author: Your Name
# Objective: Forecast next 14 days of heart rate using Prophet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os

# -------------------------------
# Step 1: Create sample data (60 days of daily average heart rate)
# -------------------------------
np.random.seed(42)
days = pd.date_range(start="2025-10-04", periods=60)
heart_rate = 70 + 0.05*np.arange(60) + 2*np.sin(np.arange(60)/5) + np.random.normal(0, 1.5, 60)
df = pd.DataFrame({"ds": days, "y": heart_rate})

# -------------------------------
# Step 2: Fit Prophet model
# -------------------------------
model = Prophet(daily_seasonality=True)
model.fit(df)

# -------------------------------
# Step 3: Forecast next 14 days
# -------------------------------
future = model.make_future_dataframe(periods=14)
forecast = model.predict(future)

# -------------------------------
# Step 4: Visualization
# -------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(out_dir, exist_ok=True)

# Save numeric forecast values to CSV
forecast_csv = os.path.join(out_dir, "forecast_values.csv")
# keep only relevant columns
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_csv, index=False)

# Save main forecast plot
fig1 = model.plot(forecast)
fig1.savefig(os.path.join(out_dir, "forecast.png"))
plt.close(fig1)

# Save trend & seasonality components
fig2 = model.plot_components(forecast)
fig2.savefig(os.path.join(out_dir, "components.png"))
plt.close(fig2)

# -------------------------------
# Step 5: Calculate MAE (training performance)
# -------------------------------
y_true = df["y"]
y_pred = forecast.loc[:59, "yhat"]   # compare first 60 points
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error (Training): {mae:.2f} bpm")

# -------------------------------
# Step 6: Extract forecast for Day 67
# -------------------------------
day_67 = forecast.iloc[66]   # since index starts from 0
print("\nForecast for Day 67:")
print(f"Date: {day_67['ds']}")
print(f"Predicted Heart Rate: {day_67['yhat']:.2f} bpm")
print(f"Confidence Interval: [{day_67['yhat_lower']:.2f}, {day_67['yhat_upper']:.2f}] bpm")

# Write summary to outputs
summary_path = os.path.join(out_dir, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
	f.write(f"Mean Absolute Error (Training): {mae:.2f} bpm\n")
	f.write("\nForecast for Day 67:\n")
	f.write(f"Date: {day_67['ds']}\n")
	f.write(f"Predicted Heart Rate: {day_67['yhat']:.2f} bpm\n")
	f.write(f"Confidence Interval: [{day_67['yhat_lower']:.2f}, {day_67['yhat_upper']:.2f}] bpm\n")

# -------------------------------
# Step 7: Differentiate forecasted heart rate (discrete daily derivative)
# -------------------------------
# We compute the day-to-day change in the predicted heart rate (yhat)
deriv = forecast[['ds', 'yhat']].copy()
deriv['yhat_diff'] = deriv['yhat'].diff()

# Save derivative to CSV
deriv_csv = os.path.join(out_dir, 'forecast_derivative.csv')
deriv.to_csv(deriv_csv, index=False)

# Plot derivative (change per day)
fig3, ax = plt.subplots(figsize=(8,4))
ax.plot(deriv['ds'], deriv['yhat_diff'], marker='o')
ax.axhline(0, color='gray', linewidth=0.7)
ax.set_title('Daily Change in Predicted Heart Rate (yhat_diff)')
ax.set_xlabel('Date')
ax.set_ylabel('Change in bpm')
fig3.autofmt_xdate()
fig3.savefig(os.path.join(out_dir, 'forecast_derivative.png'))
plt.close(fig3)
