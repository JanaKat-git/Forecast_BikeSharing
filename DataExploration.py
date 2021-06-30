'Data Exploration of the BikeSharing data set'

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Import Data
df = pd.read_csv('TRAINDATA', parse_dates=True, index_col=0)

#Info Dataset
# datetime : hourly date + timestamp  
# season :  1 = spring, 2 = summer, 3 = fall, 4 = winter 
# holiday : whether the day is considered a holiday
# workingday : whether the day is neither a weekend nor holiday
# weather : 
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 

# temp - temperature in Celsius
# atemp - "feels like" temperature in Celsius
# humidity - relative humidity
# windspeed - wind speed
# casual - number of non-registered user rentals initiated
# registered - number of registered user rentals initiated
# count - number of total rentals

# Split the Data
df['h'] = df.index.hour
df['M'] = df.index.month
df['d'] = df.index.weekday
df['sunday'] = np.where(df['d'] == 6, 1, 0)

X = df[['temp','h','holiday','M','d','sunday']]
y = df['count']

print(type(df.index)) #check for DatetimeIndex

start = df.index.min()
end = df.index.max()

print('Data goes from ' + str(start) +' till ' + str(end))

#Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(round(df.corr(),2), annot=True)
plt.savefig('./Talk/heatmap.png', dpi=150)
plt.show()

#Check emty values and Data set
print(pd.isnull(df).sum() > 0)
print(df.info())
print(df.describe())


#Visualisation
df_d = df[['d','count']]
days_counts = df_d.groupby(['d']).mean()

plt.figure(figsize=(8,6))
plt.style.use('seaborn-darkgrid')
plt.scatter(x=range(7), y=days_counts, s=100, marker='h', c='midnightblue')
ax = plt.subplot()
plt.title('Counts per Weekday')
plt.xlabel('Weekday')
ax.set_xticks([0,1,2,3,4,5,6])
ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.ylabel('Counts')
#plt.savefig('./Talk/days.png', dpi=150)
plt.show()

#Comparison weather and count
df_weather = df[['weather','count']]
df_weather.groupby('weather').size().plot.bar()
plt.title('Counts per Weather Category')
#plt.savefig('./Talk/weather_bar.png', dpi=150)

g = df_weather.groupby('weather').mean()

plt.figure(figsize=(10,8))
plt.style.use('seaborn-darkgrid')
plt.barh(y=[1,2,3,4], width=g['count'], color = 'lightseagreen')
plt.title('Counts per Weather')
plt.xlabel('Counts(mean)')
plt.title('Avg. Counts per Weather Category')
plt.xticks(rotation=0)
ax = plt.subplot()
ax.set_yticks([1,2,3,4])
plt.subplots_adjust(bottom=0.5)
ax.set_yticklabels(['1: Clear, Few clouds, Partly cloudy',
'2: Mist + ...',
'3: Snow, Rain + Thunderstorm',
'4: Extreme'])
plt.ylabel('Weather')
#plt.savefig('./Talk/weather.png', dpi=150, bbox_inches='tight')
plt.show()


#Comparision season and count
df_season = df[['season','count']]
df_season.groupby('season').size().plot.bar()
plt.title('Counts per Season')
#plt.savefig('./Talk/season_bar.png', dpi=150)
g = df_season.groupby('season').mean()

plt.figure(figsize=(10,8))
plt.style.use('bmh')
plt.barh(y=[1,2,3,4], width=g['count'], color='indigo')
plt.title('Counts per Season')
plt.xlabel('Counts(mean)')
plt.title('Avg. Counts per Season')
plt.xticks(rotation=0)
ax = plt.subplot()
ax.set_yticks([1,2,3,4])
plt.subplots_adjust(bottom=0.5)
ax.set_yticklabels(['1: Spring',
'2: Summer',
'3: Fall',
'4: Winter'])
plt.ylabel('Season')
#plt.savefig('./Talk/season.png', dpi=150, bbox_inches='tight')
plt.show()



#Comparision workingday and count
df_wday = df[['workingday','count']]
df_wday.groupby('workingday').size().plot.bar()
plt.title('Counts per Workingday')
#plt.savefig('./Talk/wday_bar.png', dpi=150)

g = df_wday.groupby('workingday').mean()
g['count'].iloc[1]

plt.figure(figsize=(8,6))
plt.style.use('bmh')
plt.bar(x=[0,0.4], height=g['count'], width=0.2, color='skyblue', edgecolor='darkcyan')
plt.title('Counts per Workingday')
plt.ylabel('Counts(mean)')
plt.yticks(rotation=0)
plt.title('Counts per "No Workingday" and Workingday')
ax = plt.subplot()
ax.set_xticks([0,0.4])
ax.set_xticklabels(['No Workingday',
'Workingday'])
plt.annotate(g['count'].iloc[0],xy=(0,g['count'].iloc[0]))
plt.annotate(g['count'].iloc[1],xy=(0.4,g['count'].iloc[1]))
plt.ylabel('Workingday')
#plt.savefig('./Talk/wday.png', dpi=150, bbox_inches='tight')
plt.show()

#ScatterPlots
y= df['count']

for x in df:
    plt.scatter(df[x],y)
    plt.xlabel(x)
    plt.ylabel('counts')
    plt.title('counts per ' + x)
    plt.show()

#Analyze different times during day
df['h'] = df.index.hour

#Divide hours in three different parts
h0_h7 = df[(df['h'] >=0) & (df['h'] <= 7)] 
h8_h16 = df[(df['h'] >7) & (df['h'] <= 16)]
h17_h23 = df[(df['h'] >16) & (df['h'] <= 23)]

plt.figure(figsize=(10,8))
ax = plt.subplot()
plt.bar(h0_h7['h'], h0_h7['count'])
plt.bar(h8_h16['h'], h8_h16['count'])
plt.bar(h17_h23['h'], h17_h23['count'])
ax.set_xticks(range(23))
plt.xlabel('hour')
plt.ylabel('count')
plt.title('Counts per hour')
#plt.savefig('./Talk/days.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,8))
ax = plt.subplot()
plt.boxplot(df['h'],vert = False)
ax.set_xticks(range(23))
plt.xlabel('hour')
plt.title('Boxplot: counts per hour')
plt.show()

#Analyze Mean
print(df.mean())

#Analyze holiday
lst_delta_no_hol = []
lst_delta_hol = []
for i in range(24):
    mean = (df['count'][(df['h'] == i)]).mean()
    no_hol = (df['count'][(df['h'] == i) & (df['holiday'] == 0)]).mean()
    hol = (df['count'][(df['h'] == 0) & (df['holiday'] == 1)]).mean() #only 13 holidays in that time
    delta1 = no_hol-mean
    lst_delta_no_hol.append(delta1)
    delta2 = hol-mean
    lst_delta_hol.append(delta2)

mean_No_holiday = (df['count'][(df['holiday'] == 0)]).mean()
mean_holiday = (df['count'][(df['holiday'] == 1)]).mean()

# (df['count'][(df['holiday'] == 1)]).index.day

print('Mean Counts for Rents during Holidays: ' + str(round((mean_No_holiday),2)))
print('Mean Counts for Rents during No_Holidays: ' + str(round((mean_holiday),2)))

#BarPlot 
plt.figure(figsize=(10,8))
plt.bar(range(24), lst_delta_no_hol, color='red')
ax = plt.subplot()
ax.set_xticks(range(24))
plt.xlabel('hour')
plt.ylabel('No_Holiday - Mean')
plt.title('Difference:"No Holiday" & Mean vs. hour ')
#plt.savefig('./Talk/no_hol.png', dpi=150, bbox_inches='tight')
plt.show

plt.figure(figsize=(10,8))
plt.bar(range(24), lst_delta_hol, color='green')
ax = plt.subplot()
ax.set_xticks(range(24))
plt.xlabel('hour')
plt.ylabel('Holiday: Delta Mean')
plt.title('Difference:Holiday & Mean vs. hour ')
plt.show


#Analyze different Months 
df['M'] = df.index.month

month_mean=[]
for month in range(1,13):
    val_mean = (df['M'] == month).mean()
    print(str(month) + ' ' + str(val_mean))
    month_mean.append(val_mean)

plt.bar(range(1,13), month_mean)
plt.xlabel('Month')
plt.ylabel('Avg. count per Month')
plt.title('Avg. Counts per Month vs Month')
plt.show()
