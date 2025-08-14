# A Computational Approach to Estimating Reconstructive Surgical Need Amidst War in Gaza

## A project by the Duke Bass Connections Team – Meeting the Need for Reconstructive Surgery in Gaza 2024-25

### Overview
This repository contains the code and datasets used to forecast injuries in Gaza during the current conflict. Our aim is to provide a transparent, reproducible model that can help estimate surgical needs both during the ongoing conflict and in its aftermath, recognizing that current infrastructure is severly limited due to the severe destruction. 
  
### Who We Are
We are part of the [Duke Bass Connections Team - Meeting the Need for Reconstructive Surgery in Gaza](https://bassconnections.duke.edu/project/meeting-need-reconstructive-surgery-palestine-2024-2025/). We are made up of undergraduate, graduate, and postdoctoral researchers, with supervision from academic physicians and faculty who have extensive experience in conflicts in the Middle East and reconstructive surgery at Duke University, Durham, NC, USA. Any views or statements made herein are our own and do not represent the institution.  

### Why This Project
The ongoing conflict in Gaza has caused extensive Palestinian civilian injuries in a setting where access to specialized reconstructive care was already limited. Damage to hospitals, loss of trained personnel, and disrupted supply chains have compounded these challenges. Recognizing these critical gaps, we aims to develop a computational model that can be used to predict evolving reconstructive surgery needs in Gaza. The future resolution or escalation of the conflict remains unclear, and as such the model is meant to be used to forecast reconstructive burdens based on multiple scenarios of possible conflict evolution.

This project does not seek to assign blame or take a political position. Our focus is entirely humanitarian: to strengthen eveidence-based planning for care delivery under extreme conditioins.

### What We Aim to Do
- Model injury incidence using injury reports, attack characterization, satellite-based infrastructure damage data, and independent news sources/social media. 
- Estimate anticipated reconstructive surgery needs for current and future conflict scenarios
- Offer a transferrable framework adaptable to other conflict or disaster contexts

### Contents
- daily_df_code includes the code to process and store the daily injury/attack data into a dataframe for analysis
- modeling_code includes the code used to build our negative binomial model
- cumulative_injuries.xlsx includes reported cumulative injuries by date as reported by OCHA_oPt, who in trun reports the data is from the Palestinian Ministry of Health
- ACLED_May_09_25_Gaza.xlsx contains data downloaded from the Armed Conflict Location and Event Data (ACLED) that details the date of attacks, attack type, location of attack (governorate, city, location, and latitude and longitude), number of fatalities, and the time stamp of each attack. ACLED is a non-profit data collection initiative that focuses on political violence across the globe. We added a column called location_classification where we separately assigned location of the attack based on "location" reported in the dataset and assigned "NA" to other locations that did not fall into the top 20 most populous municipalities/localities (reported by OCHA/PCBS in Sept 2023) or Al-Mawasi.
- pop_inf_data.xlsx contains the population density of the top 20 most populous municipalities/localities in Gaza and Al-Mawasi, as reported by the MoH, which was updated monthly based on information collected from various sources, including satellite imagery, MoH/OCHA reports, newspaper articles, and social media. Population density was classified as Low (5000 people/km^2), Medium (5000-20,000 people/km^2), and High (20,000 people/km^2). It also contains the infrastructure of a given location based on satellite imagery and MoH/OCHA reports. Infrastructure status of a given location was classified as urban, suburban, rubble, rural, or refugee camp/tent. These categories were determined based on satellite imagery, articles from independent journalists, and damage analysis reports. Refugee camps are official camps that are recognized by the UN Relief and Works Agency for Palestine Refugees in the Near East [UNWRA]).
- sources.docx includes a list of the articles and reports used to inform population and infrastructure classification. 

###
