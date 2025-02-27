# Bass Connections - Meeting the Need for Reconstructive Surgery in Palestine


Injury Datafram (injury_df)
- Example dataframe that was used to test the model. Contains the date, cumulative number of injuries in Gaza, and daily number of injuries in Gaza.
  
OCHA data
- Contains information on the total cumulative injuries in Gaza as per the OCHA daily reports (data sourced from the Ministry of Health) and the number of functioning hospitals. The number of daily injuries was manually determined by taking the difference between reported days. The number of daily injuries were also aggregated over a span of 7-12 days and a difference was then taken to get the weekly number of injuries. 

Armed Conflict Location and Event Data (ACLED) data (ACLED_data)
- Contains information collected from ACLED that details the date of attacks, event type, region of attack (govenorates, city, and latitide and longitude), number of fatalities, and the time stamp of each attack. 

Population Data
- Contains monthly population information collected from various sources, including satellite imagery and situation reports, to determine population density of the top 20 camps, cities, and townships in Gaza. For locations exceeding 20,000 people/km^2, they were classified as high density. Locations between 20,000 and 5,000 people/km^2 were medium density. Low density locations had under 5000 people/km^2. 

Infrastructure Data
- Contains monthly infrastructure information from satellite imagery and situation reports. A similar method was used to classify the same 20 locations as either urban, suburbam, rural, or barren. 


