# Bass Connections - Meeting the Need for Reconstructive Surgery in Palestine


Injury Dataframe (injury_df)
- Example dataframe that was used to test the model. Contains the date, cumulative number of injuries in Gaza, and daily number of injuries in Gaza.
  
OCHA data
- Contains information on the total cumulative injuries in Gaza as per the OCHA daily reports (data sourced from the Ministry of Health) and the number of functioning hospitals. The number of daily injuries was manually determined by taking the difference between reported days. The number of daily injuries were also aggregated over a span of 7-12 days and a difference was then taken to get the weekly number of injuries. 

Armed Conflict Location and Event Data (ACLED) data (ACLED_data)
- Contains information collected from ACLED that details the date of attacks, event type, region of attack (govenorates, city, and latitide and longitude), number of fatalities, and the time stamp of each attack. 

Population Data
- Contains monthly population information collected from various sources, including satellite imagery and situation reports, to determine population density of the top 20 camps, cities, and townships in Gaza. For locations exceeding 20,000 people/km^2, they were classified as high density. Locations between 20,000 and 5,000 people/km^2 were medium density. Low density locations had under 5000 people/km^2. 

Infrastructure Data
- Contains monthly infrastructure information from satellite imagery and situation reports. A similar method was used to classify the same 20 locations as either urban, suburbam, rural, or barren. 


# Code
1. Load Data
 ``` ruby
acled_df = pd.read_excel('/Users/el266/Documents/Professional/Duke Bass Connections/ACLED_OCT_11_UPDATED_with_dates_Gaza.xlsx', 
                         usecols=['event_date', 'sub_event_type', 'latitude', 'longitude', 'fatalities', 'admin2'])
population_data = pd.read_excel('/Users/el266/Documents/Professional/Duke Bass Connections/urban_city_data_over_time.xlsx', sheet_name= 'Population')
injury_df = pd.read_excel('/Users/el266/Documents/Professional/Duke Bass Connections/attacks_injuries.xlsx',usecols = ['date', 'number of injuries'])
infrastructure_df = pd.read_excel('/Users/el266/Documents/Professional/Duke Bass Connections/urban_city_data_over_time.xlsx', sheet_name= 'Infrastructure_2')
acled_df_2 = pd.read_excel('/Users/el266/Documents/Professional/Duke Bass Connections/ACLED_JAN_17_25_UPDATED_with_dates_Gaza.xlsx', 
                         usecols=['event_date', 'sub_event_type', 'latitude', 'longitude', 'fatalities', 'admin2'])
cumulative = pd.read_excel('/Users/el266/Documents/Professional/Duke Bass Connections/cumulative_injuries.xlsx',
                          usecols=['Total # of Injuries in Gaza (cumulative from 10/7/23)', 'Date'])
 ```
2. Data Processing
 ``` ruby
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
acled_df_2['attack_category'] = acled_df_2['sub_event_type'].map(attack_type_mapping)


# -------------------------------------------------------------------- #
# 2.1.2 Pre-processing Event Dates
# -------------------------------------------------------------------- #
acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])    # ensure 'event_date' is in datetime format
acled_df['year_month'] = acled_df['event_date'].dt.to_period('M')  # for monthly grouping purpose
 ```

3. Population Density and Infrastructure Data Assignment to Each Attack
 ``` ruby
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
    
    # Always use October 2024 for population density and infrastructure lookup if attack_date > Oct 2024
    lookup_month = '2024-10' if attack_date > pd.Timestamp('2024-10-31') else attack_date.strftime('%Y-%m')

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
        return density, 'Rural'  # Default to 'Barren' if no infrastructure info

    # Extract the infrastructure type for the specific month
    try:
        infrastructure_type = infrastructure_row[lookup_month].values[0]
    except KeyError:
        infrastructure_type = 'Rural'  # Default to 'Barren' if the month is missing
    except IndexError:
        infrastructure_type = 'Rural'  # Handle case where no matching row exists
    return density, infrastructure_type


# Iterate through each row in acled_df to calculate population density and infrastructure for each attack
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

# acled_daily.to_excel('/Users/el266/Downloads/acled_daily_agg_12_4_24.xlsx', index=False)
print('ACLED_DF:')
display(acled_df)

# Iterate through each row in acled_df_2
for idx, row in acled_df_2.iterrows():
    attack_date = row['event_date']
    attack_lat = row['latitude']
    attack_lon = row['longitude']
        
    # Use the updated function to get the area info for the current attack with October defaults
    population_density, infrastructure_type = get_area_info_for_attack(
        attack_date, attack_lat, attack_lon, population_data, infrastructure_df, vicinity_radius=1.0)
        
    # Assign results
    acled_df_2.at[idx, 'population_density'] = population_density
    acled_df_2.at[idx, 'infrastructure'] = infrastructure_type
    
print('ACLED_DF_2:')
display(acled_df_2)

population_density_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3
}

# Apply the mapping to the population_density column to both acled_df and acled_df_2
acled_df['population_density'] = acled_df['population_density'].map(population_density_mapping)
acled_df_2['population_density'] = acled_df_2['population_density'].map(population_density_mapping)
 ```
#4. Attack and Injury Data Aggregation by Date 
 ``` ruby
# Ensure injury_df date column is in the same datetime format as aclef_df
injury_df['date'] = pd.to_datetime(injury_df['date']).dt.tz_localize(None) # Remove timezone

# Merge acled_df with injury_df on the date columns
acled_with_injuries = pd.merge(acled_df, injury_df, left_on='event_date', right_on='date', how='left')

# Aggregate acled_with_injuries by event_date for acled_df
acled_daily = acled_with_injuries.groupby('event_date').agg(
    airdrone_attack_count=('attack_category', lambda x: (x == 'Air/drone strike').sum()),                   # Air/drone strike Attacks Count
    shelling_attack_count=('attack_category', lambda x: (x == 'Shelling/artillery/missile attack').sum()),  # Shelling Attacks Count
    ground_attack_count=('attack_category', lambda x: (x == 'Ground Attacks').sum()),                       # Ground Attacks Count
    # property_crime_count=('attack_category', lambda x: (x == 'Property Crimes').sum()),                   # Property Crimes Count
    civil_unrest_count=('attack_category', lambda x: (x == 'Civil Unrest').sum()),                          # Civil Unrest Count
    other_attack_count=('attack_category', lambda x: (x == 'Other').sum()),                                 # Other Attacks Count
    mean_population_density=('population_density', 'mean'),                                                 # Mean population density
    rubble_count=('infrastructure', lambda x: (x == 'Rubble').sum()),                                       # 'Barren' infrastructure Count
    camp_count=('infrastructure', lambda x: (x == 'Camp').sum()),                                           # 'Camp' infrastructure Count
    suburban_count=('infrastructure', lambda x: (x == 'Suburban').sum()),                                   # 'Suburban' infrastructure Count
    tent_count=('infrastructure', lambda x: (x == 'Tent').sum()),                                           # 'Tent' infrastructure Count
    urban_count=('infrastructure', lambda x: (x == 'Urban').sum()),                                         # 'Urban' infrastructure Count
    rural_count=('infrastructure', lambda x: (x == 'Rural').sum()),                                         # 'Urban' infrastructure Count
    injuries=('number of injuries', 'first')  
).reset_index()

# Aggregate acled_with_injuries by event_date acled_df2
acled_daily_2 = acled_df_2.groupby('event_date').agg(
    airdrone_attack_count=('attack_category', lambda x: (x == 'Air/drone strike').sum()),                   # Air/drone strike Attacks Count
    shelling_attack_count=('attack_category', lambda x: (x == 'Shelling/artillery/missile attack').sum()),  # Shelling Attacks Count
    ground_attack_count=('attack_category', lambda x: (x == 'Ground Attacks').sum()),                       # Ground Attacks Count
    # property_crime_count=('attack_category', lambda x: (x == 'Property Crimes').sum()),                   # Property Crimes Count
    civil_unrest_count=('attack_category', lambda x: (x == 'Civil Unrest').sum()),                          # Civil Unrest Count
    other_attack_count=('attack_category', lambda x: (x == 'Other').sum()),                                 # Other Attacks Count
    mean_population_density=('population_density', 'mean'),                                                 # Mean population density
    rubble_count=('infrastructure', lambda x: (x == 'Rubble').sum()),                                       # 'Barren' infrastructure Count
    camp_count=('infrastructure', lambda x: (x == 'Camp').sum()),                                           # 'Camp' infrastructure Count
    suburban_count=('infrastructure', lambda x: (x == 'Suburban').sum()),                                   # 'Suburban' infrastructure Count
    tent_count=('infrastructure', lambda x: (x == 'Tent').sum()),                                           # 'Tent' infrastructure Count
    urban_count=('infrastructure', lambda x: (x == 'Urban').sum()), 
    rural_count=('infrastructure', lambda x: (x == 'Rural').sum()), # 'Urban' infrastructure Count

).reset_index()

print('ACLED_DAILY:')
display(acled_daily)
print(acled_daily['injuries'].sum())
print(acled_daily['rubble_count'].sum())
print(acled_daily['camp_count'].sum())
print(acled_daily['suburban_count'].sum())
print(acled_daily['tent_count'].sum())
print(acled_daily['urban_count'].sum())
print(acled_daily['rural_count'].sum())
print('ACLED_DAILY_2:')
display(acled_daily_2)
 ```

#5. Training the Model 
 ``` ruby
# Filter for rows within the specified date range
start_date = '2023-10-07'
end_date = '2024-04-01'
acled_filtered = acled_daily[(acled_daily['event_date'] >= start_date) & (acled_daily['event_date'] <= end_date)]
acled_filtered = acled_filtered.dropna()


# -------------------------------------------------------------------- #
# Data Splitting into Features and Target
# -------------------------------------------------------------------- #

# Define features and target variable
X = acled_filtered[['airdrone_attack_count', 'shelling_attack_count', 'ground_attack_count', 'civil_unrest_count',
                    'mean_population_density', 'rubble_count','camp_count', 'suburban_count', 'tent_count', 'urban_count', 'rural_count']]
y = acled_filtered['injuries']
import pandas as pd
from itertools import combinations

# Center the columns of X before defining interaction terms 
X_centered = X - X.mean()

# Manually specify interaction terms
interaction_terms = pd.DataFrame({
    'airdrone_x_rubble': X_centered['airdrone_attack_count'] * X_centered['rubble_count'],
    'airdrone_x_camp': X_centered['airdrone_attack_count'] * X_centered['camp_count'],
    'airdrone_x_rural': X_centered['airdrone_attack_count'] * X_centered['rural_count'],
    'airdrone_x_suburban': X_centered['airdrone_attack_count'] * X_centered['suburban_count'],
    'airdrone_x_urban': X_centered['airdrone_attack_count'] * X_centered['urban_count'],
    'airdrone_x_tent': X_centered['airdrone_attack_count'] * X_centered['tent_count'],
    'shelling_x_camp': X_centered['shelling_attack_count'] * X_centered['camp_count'],
    'shelling_x_rubble': X_centered['shelling_attack_count'] * X_centered['rubble_count'],
    'shelling_x_rural': X_centered['shelling_attack_count'] * X_centered['rural_count'],
    'shelling_x_suburban': X_centered['shelling_attack_count'] * X_centered['suburban_count'],
    'shelling_x_urban': X_centered['shelling_attack_count'] * X_centered['urban_count'],
    'shelling_x_tent': X_centered['shelling_attack_count'] * X_centered['tent_count'],
    'ground_x_suburban': X_centered['ground_attack_count'] * X_centered['suburban_count'],
    'ground_x_urban': X_centered['ground_attack_count'] * X_centered['urban_count'],
    'ground_x_camp': X_centered['ground_attack_count'] * X_centered['camp_count'],
    'ground_x_rubble': X_centered['ground_attack_count'] * X_centered['rubble_count'],
    'ground_x_rural': X_centered['ground_attack_count'] * X_centered['rural_count'],
    'ground_x_tent': X_centered['ground_attack_count'] * X_centered['tent_count'],
    'civil_x_suburban': X_centered['civil_unrest_count'] * X_centered['suburban_count'],
    'civil_x_urban': X_centered['civil_unrest_count'] * X_centered['urban_count'],
    'civil_x_rubble': X_centered['civil_unrest_count'] * X_centered['rubble_count'],
    'civil_x_rural': X_centered['civil_unrest_count'] * X_centered['rural_count'],
    'civil_x_camp': X_centered['civil_unrest_count'] * X_centered['camp_count'],
    'civil_x_tent': X_centered['civil_unrest_count'] * X_centered['tent_count']
    # 'other_x_suburban': X_centered['other_attack_count'] * X_centered['suburban_count'],
    # 'other_x_urban': X_centered['other_attack_count'] * X_centered['urban_count'],
    # 'other_x_barren': X_centered['other_attack_count'] * X_centered['barren_count'],
    # 'other_x_camp': X_centered['other_attack_count'] * X_centered['camp_count'],
    # 'other_x_tent': X_centered['other_attack_count'] * X_centered['tent_count'],
})

# Combine original features and manually defined interaction terms
X_with_interactions = pd.concat([X_centered, interaction_terms], axis=1)

# Remove constant columns (no variance)
constant_columns = [col for col in X_with_interactions.columns if X_with_interactions[col].nunique() == 1]
X_with_interactions = X_with_interactions.drop(columns=constant_columns)
# X_with_interactions = X_with_interactions.drop(columns=['airdrone_attack_count', 'shelling_attack_count', 'ground_attack_count', 'civil_unrest_count'])
#VIF calculation
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Don't need a constant column, I think the variance_inflation_factor function adds this automatically. 
# X_with_interactions['const'] = 1

# Function to calculate VIF
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

# Function to drop features with high VIF iteratively
def drop_high_vif_features(data, threshold=10):
    while True:
        vif_data = calculate_vif(data)
        max_vif = vif_data["VIF"].max()  # Find the maximum VIF
        if max_vif < threshold:  # Stop if all VIFs are below the threshold
            break
        # Drop the feature with the highest VIF
        feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        print(f"Dropping feature: {feature_to_drop} (VIF = {max_vif})")
        data = data.drop(columns=[feature_to_drop])
    return data

# Drop features with high VIF
X_filtered = drop_high_vif_features(X_with_interactions, threshold=15)

# Calculate final VIF
final_vif_data = calculate_vif(X_filtered)
display(final_vif_data)


# -------------------------------------------------------------------- #
# Feature Selection Using LARS 
# -------------------------------------------------------------------- #

# Fit LASSO
model_lars = Lasso(alpha=0.01)
model_lars.fit(X_filtered, y)

# Get selected features
selected_features = X_filtered.columns[model_lars.coef_ != 0]
selected_coefficients = model_lars.coef_[model_lars.coef_ != 0]

# Store selected features in a DataFrame
lars_results = pd.DataFrame({'Feature': selected_features, 'LARS_Coefficient': selected_coefficients})
display(lars_results)

# # # -------------------------------------------------------------------- #
# # # Filtering the Features Selected by LARS
# # # -------------------------------------------------------------------- #

# Fit Negative Binomial GLM only on selected features from LARS
X_selected = X_filtered[selected_features]  # Use only features selected by LARS
X_const = sm.add_constant(X_selected)  # Add intercept for GLM

# # # -------------------------------------------------------------------- #
# # # Negative Binomial Generalized Linear Model
# # # -------------------------------------------------------------------- #

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

# Fit Negative Binomial model using MLE to estimate alpha
nb_model = NegativeBinomial(y, X_const).fit()

# Print model summary to get estimated alpha
print(nb_model.summary())

# Extract estimated alpha
alpha_hat = nb_model.params[-1]  # The last parameter in the model is alpha
print(f"\nEstimated Alpha: {alpha_hat}")

glm_nb_model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial(alpha=alpha_hat)).fit()
print(glm_nb_model.summary())

# Get coefficients and p-values from GLM
glm_results = pd.DataFrame({
    'Feature': X_const.columns,  # Get feature names (including intercept)
    'GLM_NB_Coefficient': glm_nb_model.params.values,  
    'p-value': glm_nb_model.pvalues.values  
})


# Drop the intercept ('const') from glm_results before merging
glm_results = glm_results[glm_results['Feature'] != 'const']

#Identify significant variables (e.g., p < 0.05)
significant_vars = glm_results[glm_results['p-value'] < 0.05]['Feature'].tolist()

#Subset the dataset to keep only the significant variables
X_significant = sm.add_constant(X_const[significant_vars])

print(X_significant)


#Refit the Negative Binomial GLM
glm_nb_model_significant = sm.GLM(y, X_significant, family=sm.families.NegativeBinomial(alpha=alpha_hat)).fit()

#Print new model summary
print(glm_nb_model_significant.summary())


#Extract coefficients and p-values from the new model
glm_results_significant = pd.DataFrame({
    'Feature': X_significant.columns,  
    'GLM_NB_Coefficient': glm_nb_model_significant.params.values,  
    'p-value': glm_nb_model_significant.pvalues.values  
})

# Print the refined results
print("\nRefined Model Results (Only Significant Predictors):")
display(glm_results_significant)
 ```

#6. Projections 
 ``` ruby
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------------------------------------------------------- #
# Prediction and Sensitivity Analysis
# -------------------------------------------------------------------- #

# Extract coefficients, ensuring they are indexed by feature names
glm_coeffs = glm_results_significant.set_index("Feature")["GLM_NB_Coefficient"]

# Ensure X_const only includes significant predictors
X_significant_only = X_const[glm_coeffs.index]

# Compute baseline predicted injuries
predicted_injuries = np.exp(X_significant_only.dot(glm_coeffs))

# Apply ±10% variation to independent variables (excluding intercept)
X_high = X_significant_only.copy()
X_low = X_significant_only.copy()

X_high.iloc[:, 1:] *= 1.1  # Increase all non-intercept variables by 10%
X_low.iloc[:, 1:] *= 0.9   # Decrease all non-intercept variables by 10%

# Compute new predictions with varied inputs
predicted_upper = np.exp(X_high.dot(glm_coeffs))
predicted_lower = np.exp(X_low.dot(glm_coeffs))

# -------------------------------------------------------------------- #
# Cumulative Injuries Prediction
# -------------------------------------------------------------------- #

# Ensure event_date is in datetime format and align with actual injuries
results_df = pd.DataFrame({
    'event_date': pd.to_datetime(acled_filtered['event_date']),
    'actual_injuries': y.values,
    'predicted_injuries': predicted_injuries.values,
    'predicted_upper': predicted_upper.values,
    'predicted_lower': predicted_lower.values
})

# Sort by event_date
results_df = results_df.sort_values('event_date')

# Compute cumulative sum of injuries
results_df['cumulative_actual'] = results_df['actual_injuries'].cumsum()
results_df['cumulative_predicted'] = results_df['predicted_injuries'].cumsum()
results_df['cumulative_upper'] = results_df['predicted_upper'].cumsum()
results_df['cumulative_lower'] = results_df['predicted_lower'].cumsum()

# Convert event_date to monthly periods for aggregation
results_df['month'] = results_df['event_date'].dt.to_period('M')

# -------------------------------------------------------------------- #
# Visualization
# -------------------------------------------------------------------- #

# Plot cumulative actual vs. predicted injuries
plt.figure(figsize=(14, 8))

plt.plot(results_df['event_date'], results_df['cumulative_actual'], label="Cumulative Actual Injuries", 
         color='black', linewidth=2)
plt.plot(results_df['event_date'], results_df['cumulative_predicted'], label="Cumulative Predicted Injuries", 
         color='blue', linestyle="--", linewidth=2)
plt.fill_between(results_df['event_date'], results_df['cumulative_lower'], results_df['cumulative_upper'], 
                 color='blue', alpha=0.2, label="10% Sensitivity Range")

# Format x-axis to show each month
plt.xlabel("Time (Months)")
plt.ylabel("Cumulative Injuries")
plt.title("Cumulative Actual vs. Predicted Injuries with Sensitivity Analysis")

# Set x-axis ticks to the first day of each month
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as "Jan 2024", "Feb 2024", etc.

plt.xticks(rotation=0)
plt.legend()
plt.grid(False)
# plt.savefig('/Users/el266/Downloads/LARS_GLM_daily_first_few.png', dpi=350)
plt.show()

# -------------------------------------------------------------------- #
# Monthly Aggregation
# -------------------------------------------------------------------- #

# Aggregate injuries by month
monthly_results = results_df.groupby('month').agg({
    'actual_injuries': 'sum', 
    'predicted_injuries': 'sum',
    'predicted_upper': 'sum',
    'predicted_lower': 'sum'
}).reset_index()

# Compute cumulative injuries over months
monthly_results['cumulative_actual'] = monthly_results['actual_injuries'].cumsum()
monthly_results['cumulative_predicted'] = monthly_results['predicted_injuries'].cumsum()
monthly_results['cumulative_upper'] = monthly_results['predicted_upper'].cumsum()
monthly_results['cumulative_lower'] = monthly_results['predicted_lower'].cumsum()

# Display the final monthly results
display(monthly_results)
 ```

