import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

rand_seed=1024
data = pd.read_csv('~/tmp/flights.csv')
# cols: date,flight_id,route,aircraft_type,scheduled_hour,crew_duty_hours,weather_forecast_18h,
#       weather_actual,atc_slot_delay_min,delayed
# print(data.isna().sum())
# print(data.duplicated().sum())
# print(data.dtypes)

# sns.catplot(x="delayed", kind="count", data=data)
# plt.show()

data = data.sort_values(by="date", ascending=True)
train_data = data[data["date"]<'2026-07-03']
test_data = data[data["date"]>'2026-07-02']
X_train, X_test = train_data.iloc[:, :-1],test_data.iloc[:, :-1]
y_train, y_test = train_data.iloc[:, -1],test_data.iloc[:, -1]

test_data.to_csv("../data/test_data.csv")
