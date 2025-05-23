# -*- coding: utf-8 -*-



from acled_analysis import load_data, process_attacks, aggregate_acled

# Example file paths (user can modify as needed)
acled_path = 'data/ACLED_OCT_11_UPDATED_with_dates_Gaza.xlsx'
acled2_path = 'data/ACLED_JAN_17_25_UPDATED_with_dates_Gaza.xlsx'
pop_path = 'data/urban_city_data_over_time.xlsx'
infra_path = 'data/urban_city_data_over_time.xlsx'
injury_path = 'data/attacks_injuries.xlsx'
cumulative_path = 'data/cumulative_injuries.xlsx'


acled_df, pop_data, injury_df, infra_df, acled_df_2, cumulative = load_data(
    acled_path, pop_path, injury_path, infra_path, acled2_path, cumulative_path)


acled_df = process_attacks(acled_df, pop_data, infra_df)
acled_df_2 = process_attacks(acled_df_2, pop_data, infra_df)

acled_daily = aggregate_acled(acled_df, injury_df)
acled_daily_2 = aggregate_acled(acled_df_2)

print(acled_daily.head())
print("Total injuries:", acled_daily['injuries'].sum())


