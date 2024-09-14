# Import required packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm

pd.options.mode.chained_assignment = None

# Load the cleaned dataset
city_temps_sarima = pd.read_csv("./data/city_temps_cleaned.csv")
arhus = city_temps_sarima[city_temps_sarima.City == "Århus"]
dates = arhus[["dt", "AverageTemperature"]]
dates["dt"] = pd.to_datetime(dates["dt"], format="%Y-%m-%d")
dates.set_index("dt", inplace=True)
dates.index = pd.DatetimeIndex(dates.index.values, freq=dates.index.inferred_freq)

# Train the SARIMA model
sarima_model = sm.tsa.statespace.SARIMAX(dates, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
result = sarima_model.fit()

# Plot the model diagnostics
result.plot_diagnostics(figsize=(16, 12))

# Predict surface temperatures for the next four years
forecast = result.get_forecast(steps=48)
forecast.summary_frame()

# Plot the surface temperature forecast
last_few_years = dates.tail(360)

ax1 = last_few_years.plot()
forecast.predicted_mean.plot(ax=ax1, label="Forecast")

ci = forecast.conf_int()
ax1.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color="gray", alpha=0.2)

ax1.set_xlabel("Time")
ax1.set_ylabel("Temperature (C)")
ax1.set_title("Temperature Forecast with SARIMA")

# Evaluate the performance of the model
pred_y = result.predict(start="1743-11-01", end="2013-09-01")

mseSARIMA = mean_squared_error(dates, pred_y)
print(mseSARIMA)
