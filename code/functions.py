# -*- coding: utf-8 -*-

import pandas as pd
import geopy.distance


ATTACK_CATEGORIES = {
    'Air/drone strike': ['Air/drone strike'],
    'Ground Attacks': ['Armed clash', 'Attack', 'Remote explosive/landmine/IED', 'Grenade', 'Change to group/activity'],
    'Shelling/artillery/missile attack': ['Shelling/artillery/missile attack'],
    'Civil Unrest': ['Mob violence', 'Looting/property destruction', 'Violent demonstration', 'Peaceful protest',
                     'Disrupted weapons use', 'Excessive force against protesters', 'Arrests',
                     'Protest with intervention', 'Abduction/forced disappearance', 'Sexual violence'],
    'Other': ['Other']
}
ATTACK_TYPE_MAPPING = {event: category for category, events in ATTACK_CATEGORIES.items() for event in events}
POP_DENSITY_MAP = {'Low': 1, 'Medium': 2, 'High': 3}


def load_data(acled_path, pop_path, injury_path, infra_path, acled2_path, cumulative_path):
    acled_df = pd.read_excel(acled_path, usecols=['event_date', 'sub_event_type', 'latitude', 'longitude', 'fatalities', 'admin2'])
    population_data = pd.read_excel(pop_path, sheet_name='Population')
    injury_df = pd.read_excel(injury_path, usecols=['date', 'number of injuries'])
    infrastructure_df = pd.read_excel(infra_path, sheet_name='Infrastructure_2')
    acled_df_2 = pd.read_excel(acled2_path, usecols=['event_date', 'sub_event_type', 'latitude', 'longitude', 'fatalities', 'admin2'])
    cumulative = pd.read_excel(cumulative_path, usecols=['Total # of Injuries in Gaza (cumulative from 10/7/23)', 'Date'])
    return acled_df, population_data, injury_df, infrastructure_df, acled_df_2, cumulative


def haversine(lat1, lon1, lat2, lon2):
    return geopy.distance.distance((lat1, lon1), (lat2, lon2)).km


def is_attack_in_vicinity(attack_lat, attack_lon, area_data, radius):
    return any(haversine(attack_lat, attack_lon, row['latitude'], row['longitude']) <= radius for _, row in area_data.iterrows())


def find_closest_location(attack_lat, attack_lon, area_data):
    return min(area_data.index, key=lambda idx: haversine(attack_lat, attack_lon, area_data.at[idx, 'latitude'], area_data.at[idx, 'longitude']))


def get_area_info(attack_date, attack_lat, attack_lon, pop_data, infra_df, radius):
    month = '2024-10' if attack_date > pd.Timestamp('2024-10-31') else attack_date.strftime('%Y-%m')
    if not is_attack_in_vicinity(attack_lat, attack_lon, pop_data, radius):
        return 'Low', 'Rural'
    closest_idx = find_closest_location(attack_lat, attack_lon, pop_data)
    pop_row = pop_data.loc[closest_idx]
    density = pop_row.get(month, 'Low')
    infra_row = infra_df[infra_df['location'] == pop_row['location']]
    infra_type = 'Rural'
    if not infra_row.empty:
        try:
            infra_type = infra_row[month].values[0]
        except (KeyError, IndexError):
            pass
    return density, infra_type


def process_attacks(acled_df, pop_data, infra_df, radius=1.0):
    acled_df['attack_category'] = acled_df['sub_event_type'].map(ATTACK_TYPE_MAPPING)
    acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])
    acled_df['year_month'] = acled_df['event_date'].dt.to_period('M')

    for idx, row in acled_df.iterrows():
        density, infra = get_area_info(row['event_date'], row['latitude'], row['longitude'], pop_data, infra_df, radius)
        acled_df.at[idx, 'population_density'] = density
        acled_df.at[idx, 'infrastructure'] = infra
    acled_df['population_density'] = acled_df['population_density'].map(POP_DENSITY_MAP)
    return acled_df


def aggregate_acled(acled_df, injury_df=None):
    if injury_df is not None:
        injury_df['date'] = pd.to_datetime(injury_df['date']).dt.tz_localize(None)
        acled_df = pd.merge(acled_df, injury_df, left_on='event_date', right_on='date', how='left')

    agg_dict = {
        'airdrone_attack_count': ('attack_category', lambda x: (x == 'Air/drone strike').sum()),
        'shelling_attack_count': ('attack_category', lambda x: (x == 'Shelling/artillery/missile attack').sum()),
        'ground_attack_count': ('attack_category', lambda x: (x == 'Ground Attacks').sum()),
        'civil_unrest_count': ('attack_category', lambda x: (x == 'Civil Unrest').sum()),
        'other_attack_count': ('attack_category', lambda x: (x == 'Other').sum()),
        'mean_population_density': ('population_density', 'mean'),
        'rubble_count': ('infrastructure', lambda x: (x == 'Rubble').sum()),
        'camp_count': ('infrastructure', lambda x: (x == 'Camp').sum()),
        'suburban_count': ('infrastructure', lambda x: (x == 'Suburban').sum()),
        'tent_count': ('infrastructure', lambda x: (x == 'Tent').sum()),
        'urban_count': ('infrastructure', lambda x: (x == 'Urban').sum()),
        'rural_count': ('infrastructure', lambda x: (x == 'Rural').sum())
    }
    if injury_df is not None:
        agg_dict['injuries'] = ('number of injuries', 'first')

    return acled_df.groupby('event_date').agg(**agg_dict).reset_index()
