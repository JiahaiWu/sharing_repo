import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

rand_seed=1024
data = pd.read_csv('../data/flights.csv')
# cols: date,flight_id,route,aircraft_type,scheduled_hour,crew_duty_hours,weather_forecast_18h,
#       weather_actual,atc_slot_delay_min,delayed
print(data.isna().sum())
print(data.duplicated().sum())
print(data.dtypes)

sns.catplot(x="delayed", kind="count", data=data)
plt.show()
