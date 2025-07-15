# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import distance
import geopy.distance
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Lasso
import matplotlib.dates as mdates

# -------------------------------------------------------------------- #
# 1. Load Data From Excel Files
# -------------------------------------------------------------------- #

# ACLED event data
acled_df = pd.read_excel('ACLED_May_09_25_Gaza.xlsx.xlsx', 
                         usecols=['event_date', 'sub_event_type', 'latitude', 'longitude', 'fatalities', 'admin2'])

# Population data 
population_data = pd.read_excel('urban_city_data_over_time.xlsx', sheet_name='Population')

# Infrastructure data 
infrastructure_df = pd.read_excel('urban_city_data_over_time.xlsx', sheet_name='Infrastructure_2')

# Injury data by date
injury_df = pd.read_excel('attacks_injuries.xlsx', usecols=['date', 'number of injuries'])

# Cumulative injury data 
cumulative = pd.read_excel('cumulative_injuries.xlsx', usecols=['Total # of Injuries in Gaza (cumulative from 10/7/23)', 'Date'])

acled_df_2 = pd.read_excel('ACLED_JAN_17_25_UPDATED_with_dates_Gaza.xlsx', 
                         usecols=['event_date', 'sub_event_type', 'latitude', 'longitude', 'fatalities', 'admin2'])
# -------------------------------------------------------------------- #
# 2. Map Sub-Event Types to Broad Attack Categories
# -------------------------------------------------------------------- #

attack_categories = {
    'Air/drone strike': ['Air/drone strike'],
    'Ground Attacks': ['Armed clash', 'Attack', 'Remote explosive/landmine/IED', 'Grenade', 'Change to group/activity'],
    'Shelling/artillery/missile attack': ['Shelling/artillery/missile attack'],
    'Civil Unrest': [
        'Mob violence', 'Looting/property destruction', 'Violent demonstration', 'Peaceful protest',
        'Disrupted weapons use', 'Excessive force against protesters', 'Arrests',
        'Protest with intervention', 'Abduction/forced disappearance', 'Sexual violence'
    ],
    'Other': ['Other']
}


attack_type_mapping = {event: category for category, events in attack_categories.items() for event in events}
acled_df['attack_category'] = acled_df['sub_event_type'].map(attack_type_mapping)


# -------------------------------------------------------------------- #
# 3. Format Dates and Add Time Info
# -------------------------------------------------------------------- #

acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])
acled_df['year_month'] = acled_df['event_date'].dt.to_period('M')


# -------------------------------------------------------------------- #
# 4. Helper Functions for Proximity and Lookup
# -------------------------------------------------------------------- #

def haversine(lat1, lon1, lat2, lon2):
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).km

def is_attack_in_vicinity(attack_lat, attack_lon, area_data, vicinity_radius):
    for _, row in area_data.iterrows():
        distance = haversine(attack_lat, attack_lon, row['latitude'], row['longitude'])
        if distance <= vicinity_radius:
            return True
    return False

def find_closest_location(attack_lat, attack_lon, area_data):
    min_distance = float('inf')
    closest_idx = None
    for idx, row in area_data.iterrows():
        distance = haversine(attack_lat, attack_lon, row['latitude'], row['longitude'])
        if distance < min_distance:
            min_distance = distance
            closest_idx = idx
    return closest_idx

def get_area_info_for_attack(attack_date, attack_lat, attack_lon, population_data, infrastructure_df, vicinity_radius):
    
    # lookup the month
    lookup_month = attack_date.strftime('%Y-%m')

    # Check if the attack is within the vicinity of any location
    if not is_attack_in_vicinity(attack_lat, attack_lon, population_data, vicinity_radius):
        return 'Low', 'Rural'  # Default values for low density and barren infrastructure

    # Find the closest location
    closest_idx = find_closest_location(attack_lat, attack_lon, population_data)
    if closest_idx is None:
        return 'Low', 'Rural'  # No location found, return defaults

    # Get the population density for the lookup month (location-specific)
    population_row = population_data.loc[closest_idx]
    density = population_row.get(lookup_month, 'Low')  # Default to 'Low' if no value exists for the month

    # Get the infrastructure type for the lookup month (location-specific)
    infrastructure_row = infrastructure_df[infrastructure_df['location'] == population_row['location']]
    if infrastructure_row.empty:
        return density, 'Rural'  # Default to 'Rural' if no infrastructure info

    # Extract the infrastructure type for the specific month
    try:
        infrastructure_type = infrastructure_row[lookup_month].values[0]
    except KeyError:
        infrastructure_type = 'Rural'  # Default to 'Rural' if the month is missing
    except IndexError:
        infrastructure_type = 'Rural'  # Handle case where no matching row exists
    return density, infrastructure_type


# -------------------------------------------------------------------- #
# 5. Apply Population Density and Infrastructure Info
# -------------------------------------------------------------------- #

for idx, row in acled_df.iterrows():
    # Get the attack date, latitude, and longitude
    attack_date = row['event_date']
    attack_lat = row['latitude']
    attack_lon = row['longitude']
        
    # Use the function to get the area info for the current attack
    population_density, infrastructure_type = get_area_info_for_attack(
        attack_date, attack_lat, attack_lon, population_data, infrastructure_df, vicinity_radius=1.0)
        
    # Assign the results to the corresponding columns
    acled_df.at[idx, 'population_density'] = population_density
    acled_df.at[idx, 'infrastructure'] = infrastructure_type


# -------------------------------------------------------------------- #
# 6. Map Population Density to Ordinal Values
# -------------------------------------------------------------------- #

population_density_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
acled_df['population_density'] = acled_df['population_density'].map(population_density_mapping)


# -------------------------------------------------------------------- #
# 7. Merge with Daily Injury Data
# -------------------------------------------------------------------- #

injury_df['date'] = pd.to_datetime(injury_df['date']).dt.tz_localize(None)
acled_with_injuries = pd.merge(acled_df, injury_df, left_on='event_date', right_on='date', how='left')


# -------------------------------------------------------------------- #
# 8. Aggregate Data by Day
# -------------------------------------------------------------------- #

acled_daily = acled_with_injuries.groupby('event_date').agg(
    airdrone_attack_count=('attack_category', lambda x: (x == 'Air/drone strike').sum()),
    shelling_attack_count=('attack_category', lambda x: (x == 'Shelling/artillery/missile attack').sum()),
    ground_attack_count=('attack_category', lambda x: (x == 'Ground Attacks').sum()),
    civil_unrest_count=('attack_category', lambda x: (x == 'Civil Unrest').sum()),
    other_attack_count=('attack_category', lambda x: (x == 'Other').sum()),
    mean_population_density=('population_density', 'mean'),
    rubble_count=('infrastructure', lambda x: (x == 'Rubble').sum()),
    camp_count=('infrastructure', lambda x: (x == 'Camp').sum()),
    suburban_count=('infrastructure', lambda x: (x == 'Suburban').sum()),
    tent_count=('infrastructure', lambda x: (x == 'Tent').sum()),
    urban_count=('infrastructure', lambda x: (x == 'Urban').sum()),
    rural_count=('infrastructure', lambda x: (x == 'Rural').sum()),
    injuries=('number of injuries', 'first')
).reset_index()


# -------------------------------------------------------------------- #
# 9. Save and Display Summary Stats
# -------------------------------------------------------------------- #

# Export aggregated daily data
acled_daily.to_excel("acled_daily_processed.xlsx", index=False)

# Optional summary prints
print("Summary Statistics:")
print(acled_daily['injuries'].sum())
print(acled_daily['rubble_count'].sum())
print(acled_daily['camp_count'].sum())
print(acled_daily['suburban_count'].sum())
print(acled_daily['tent_count'].sum())
print(acled_daily['urban_count'].sum())
print(acled_daily['rural_count'].sum())
