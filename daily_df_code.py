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


#Load Data
acled_df = pd.read_excel('ACLED_May_09_25_Gaza.xlsx', 
                         usecols=['event_date', 'sub_event_type', 'latitude', 'longitude', 'fatalities', 'admin2', 'location_classification'])
population_data = pd.read_excel('pop_inf_data.xlsx', sheet_name= 'Population')
injury_df = pd.read_excel('cumulative_injuries.xlsx',usecols = ['Date', 'Daily # of injuries (Gaza)'])
infrastructure_df = pd.read_excel('pop_inf_data.xlsx', sheet_name= 'Infrastructure')
cumulative = pd.read_excel('cumulative_injuries.xlsx',
                          usecols=['Date','Total # of Injuries in Gaza (cumulative from 10/7/23)'])
# -------------------------------------------------------------------- #
# 2.1.1 Mapping ACLED Sub-Event Types to Broader Attack Categories
# -------------------------------------------------------------------- #

# Define mapping of attack types to broader categories
attack_categories = {
    'Air/drone strike': ['Air/drone strike'],
    'Ground Attacks': ['Armed clash', 'Attack', 'Remote explosive/landmine/IED', 'Grenade', 
                       'Change to group/activity'],
    'Shelling/artillery/missile attack': ['Shelling/artillery/missile attack'],
    'Civil Unrest': ['Mob violence', 'Looting/property destruction', 'Violent demonstration', 'Peaceful protest', 
                     'Disrupted weapons use', 'Excessive force against protesters', 'Arrests', 
                     'Protest with intervention', 'Abduction/forced disappearance', 'Sexual violence'],
    'Other': ['Other']
}

attack_type_mapping = {event: category for category, events in attack_categories.items() for event in events}


# Map attack types to broader categories
acled_df['attack_category'] = acled_df['sub_event_type'].map(attack_type_mapping)


# -------------------------------------------------------------------- #
# 2.1.2 Pre-processing Event Dates
# -------------------------------------------------------------------- #
acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])    # ensure 'event_date' is in datetime format
acled_df['year_month'] = acled_df['event_date'].dt.to_period('M')  # for monthly grouping purpose


# -------------------------------------------------------------------------------------------- #
# 2.2.1 Helper Functions for Determining Population Density and Infrastructure Type
# -------------------------------------------------------------------------------------------- #

# Function to calculate the Haversine distance between two points (lat, lon)
def haversine(lat1, lon1, lat2, lon2):
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).km

# Function to check if the attack is within the vicinity of any location based on area radius
def is_attack_in_vicinity(attack_lat, attack_lon, area_data, vicinity_radius):
    for _, row in area_data.iterrows():
        distance = haversine(attack_lat, attack_lon, row['latitude'], row['longitude'])
        if distance <= vicinity_radius:
            return True
    return False

# Function to find the closest location for an attack if it's within vicinity
def find_closest_location(attack_lat, attack_lon, area_data):
    min_distance = float('inf')
    closest_idx = None
    for idx, row in area_data.iterrows():
        distance = haversine(attack_lat, attack_lon, row['latitude'], row['longitude'])
        if distance < min_distance:
            min_distance = distance
            closest_idx = idx
    return closest_idx

# Function to get population density and infrastructure setting for an attack
def get_area_info_for_attack(attack_date, attack_lat, attack_lon, population_data, infrastructure_df, vicinity_radius):
    
    # lookup the month
    lookup_month = attack_date.strftime('%Y-%m')

    # Check if the attack is within the vicinity of any location
    if not is_attack_in_vicinity(attack_lat, attack_lon, population_data, vicinity_radius):
        return 'Low', 'Rural', "NA"  # Default values for low density and barren infrastructure

    # Find the closest location
    closest_idx = find_closest_location(attack_lat, attack_lon, population_data)
    if closest_idx is None:
        return 'Low', 'Rural', "NA" # No location found, return defaults

    #get location for the row
    population_row = population_data.loc[closest_idx]
    location = population_row['location']
    
    # Get the population density for the lookup month (location-specific)
    density = population_row.get(lookup_month, 'Low')  # Default to 'Low' if no value exists for the month

    # Get the infrastructure type for the lookup month (location-specific)
    infrastructure_row = infrastructure_df[infrastructure_df['location'] == population_row['location']]
    if infrastructure_row.empty:
        return density, 'Rural', "NA"  # Default to 'Rural' if no infrastructure info

    # Extract the infrastructure type for the specific month
    try:
        infrastructure_type = infrastructure_row[lookup_month].values[0]
    except KeyError:
        infrastructure_type = 'Rural'  # Default to 'Rural' if the month is missing
    except IndexError:
        infrastructure_type = 'Rural'  # Handle case where no matching row exists

    return density, infrastructure_type, location

def get_area_info_for_attack_two(attack_date, classification, population_data, infrastructure_df):
    # lookup the month
    lookup_month = attack_date.strftime('%Y-%m')
    
    if classification=="NA":
        return "Low","Rural","NA"

    # Lookup population row
    population_row = population_data[population_data['location'] == classification]
    if population_row.empty:
        return "Low", "Rural", classification
    else:
        population_row = population_row.iloc[0]  # Convert to Series
        density = population_row.get(lookup_month, "Low")
    
    # Lookup infrastructure row
    infrastructure_row = infrastructure_df[infrastructure_df['location'] == classification]
    if infrastructure_row.empty:
        return "Low", "Rural", classification
    else:
        infrastructure_row = infrastructure_row.iloc[0]
        infrastructure_type = infrastructure_row.get(lookup_month, "Rural")
    
    
    
    return density, infrastructure_type, classification



# -------------------------------------------------------------------------------------------- #
# 2.2.2 Applying Population Density and Infrastructure Information to ACLED Data
# -------------------------------------------------------------------------------------------- #

# Iterate through each row in acled_df to calculate population density and infrastructure for each attack
for idx, row in acled_df.iterrows():
    # Get the attack date, latitude, and longitude
    attack_date = row['event_date']
    attack_lat = row['latitude']
    attack_lon = row['longitude']
    classification = row['location_classification']
        
    # Use the function to get the area info for the current attack
    #population_density, infrastructure_type, location = get_area_info_for_attack(attack_date, attack_lat, attack_lon, population_data, infrastructure_df, vicinity_radius=1.0)
    population_density, infrastructure_type, location = get_area_info_for_attack_two(attack_date, classification, population_data, infrastructure_df)    
    # Assign the results to the corresponding columns
    acled_df.at[idx, 'population_density'] = population_density
    acled_df.at[idx, 'infrastructure'] = infrastructure_type
    acled_df.at[idx,'location']=location


# -------------------------------------------------------------------------------------------- #
# 2.2.3 Mapping Population Density To Ordinal Values
# -------------------------------------------------------------------------------------------- #

population_density_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3
}

# # Apply the mapping to the population_density column
acled_df['population_density'] = acled_df['population_density'].map(population_density_mapping)
print(acled_df)


# -------------------------------------------------------------------- #
# 2.3.1 Fixing Date Format and Merging ACLED and Injuries Datasets 
# -------------------------------------------------------------------- #

# Ensure injury_df date column is in the same datetime format as aclef_df
injury_df['Date'] = pd.to_datetime(injury_df['Date']).dt.tz_localize(None) # Remove timezone

acled_df["group_key"] = (
    acled_df["attack_category"].str[:3].str.lower() + "-" +
    acled_df["infrastructure"].str.lower().str.replace(" ", "_") + "-pop_" +
    acled_df["population_density"].astype(str)
)



# Calculate percent of attacks per day that occurred in new zones (based on past 14 days)
percentage_new_zone_rows = []

for current_date in acled_df['event_date'].unique():
    acled_df['zone_id'] = acled_df['latitude'].round(3).astype(str) + "_" + acled_df['longitude'].round(3).astype(str)
    # Filter today's and past days' data
    today_df = acled_df[acled_df['event_date'] == current_date]
    start_date = current_date - pd.Timedelta(days=1)
    past_df = acled_df[(acled_df['event_date'] < current_date) & (acled_df['event_date'] >= start_date)]
    
    # Zones from past days
    past_zones = set(past_df['zone_id'])

    # Mark which attacks today are in new zones
    is_new = ~today_df['zone_id'].isin(past_zones)
    pct_new = is_new.sum() / len(today_df) if len(today_df) > 0 else 0
    
    percentage_new_zone_rows.append({
        'event_date': current_date,
        'pct_new_zone_attacks': pct_new
    })

# Create dataframe and merge back
percentage_df = pd.DataFrame(percentage_new_zone_rows)
acled_df = acled_df.merge(percentage_df, on='event_date', how='left')

#Aggregate by date
zone_summary = acled_df.groupby('event_date').agg(
    total_attacks=('zone_id', 'count'),
    new_zone_attacks=('pct_new_zone_attacks', 'sum'),
    avg_pct_new_zone_attacks=('pct_new_zone_attacks', 'mean')
#    avg_location_lat=('latitude','mean'),
#    avg_location_long=('longitude','mean')
).reset_index()


# -------------------------------------------------------------------- #
# 2.3.2 Data Aggregation by Event Date 
# -------------------------------------------------------------------- #

grouped = acled_df.groupby(["event_date", "group_key"]).size().reset_index(name="count")
acled_formatted = grouped.pivot(index="event_date", columns="group_key", values="count").fillna(0).astype(int)
acled_formatted = acled_formatted.reset_index()

# Merge acled_df_grouped with injury_df on the date columns
acled_daily=pd.merge(acled_formatted, injury_df, left_on='event_date', right_on='Date', how='left')
acled_daily = acled_daily.rename(columns={"Daily # of injuries (Gaza)": "injuries"})

#Merge with zone data
acled_daily = pd.merge(acled_daily, zone_summary, on='event_date', how='left')
acled_daily["pct_new_zone_attacks_14d"] = acled_daily["avg_pct_new_zone_attacks"].rolling(14).mean()
acled_daily=acled_daily[acled_daily["event_date"] >= "2023-10-01"]

acled_daily.to_excel("acled_daily.xlsx", index=False) ##change save to for your own folder




