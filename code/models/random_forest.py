# Import required packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Load the cleaned dataset
df = pd.read_csv("city_temps_cleaned.csv")

# Split dataset into training and test sets
df['Latitude-Direction'] = df['Latitude'].str[-1]
df['Longitude-Direction'] = df['Longitude'].str[-1]
df['Latitude-Value'] = pd.to_numeric(df['Latitude'].str[0:-1])
df['Longitude-Value'] = pd.to_numeric(df['Longitude'].str[0:-1])
df['dt'] = pd.to_datetime(df['dt'])
df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month.astype(str)
long_direc = pd.get_dummies(df['Longitude-Direction'])
lat_direc = pd.get_dummies(df['Latitude-Direction'])
month = pd.get_dummies(df['month'])
df = df.join([long_direc, lat_direc, month])

df = df[df['dt'] > '1900-01-01']

prep_df = df[['AverageTemperature', 'Latitude-Value', 'Longitude-Value', 'year', 'E', 'N', 'S', 'W', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']]

X_train, X_test, y_train, y_test = train_test_split(prep_df.iloc[:,1:], prep_df.iloc[:,0], test_size= 0.2, shuffle=True)

# Train the Random Forest model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
predictYrf = rf.predict(X_test)

# Evaluate the performance of the model
mseRF = mean_squared_error(y_test, predictYrf)
print(mseRF)