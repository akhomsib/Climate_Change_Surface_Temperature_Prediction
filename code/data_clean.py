import pandas as pd
import numpy as np

original_df = pd.read_csv("./data/source/GlobalLandTemperaturesByCity.csv")
clean_df = pd.DataFrame()

cities = original_df["City"].unique()
for city in cities:

    # Grab values related to city
    city_block = original_df[original_df.City == city]
    num_values = city_block.shape[0]

    # Fix null values with interpolation
    fix_temps = city_block["AverageTemperature"].interpolate()
    city_block["AverageTemperature"] = fix_temps

    # Update new df
    clean_df = pd.concat([clean_df, city_block])
    
clean_df.to_csv("./data/city_temps_cleaned.csv")

# For global temps
df = pd.read_csv("./data/source/GlobalTemperatures.csv")
fixed_land_temps = df["LandAverageTemperature"].interpolate()
df["LandAverageTemperature"] = fixed_land_temps

df.to_csv("./data/global_temps_cleaned.csv")
