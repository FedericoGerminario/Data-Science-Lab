import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

dataset = pd.read_csv('delay_data.csv', parse_dates=[0])
dataset.info()
dataset.describe()
mask = (dataset['CANCELLED'] < 1)

data = dataset.loc[mask]

#4 point
carrierData = data.groupby(['UNIQUE_CARRIER']).sum()
data['MEAN_DELAY'] = data.iloc[:, -6: -2].mean(axis=1)

#5 point
data['DayOfWeek'] = data['FL_DATE'].dt.dayofweek
data['delayDelta'] = data['DEP_DELAY'] - data['ARR_DELAY']

#6 point
data.groupby('DayOfWeek')['ARR_DELAY'].mean().plot()
#plt.show()

#7 point
mask = data['DayOfWeek'] > 5
carrierdelay = data.loc[mask].groupby('UNIQUE_CARRIER')['ARR_DELAY'].mean()
carrierarrivaltime = data.loc[~mask].groupby('UNIQUE_CARRIER')['ARR_TIME'].mean()

#8 point
multidata = data.set_index(['UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'FL_DATE'])

#9 point
print(multidata.loc[['AA', 'DL'],['LAX'],:,:][['DEP_TIME', 'DEP_DELAY']])

#10 point
mask = multidata.index.get_level_values('FL_DATE').isin([20170101, 20170102, 20170103, 20170104, 20170105, 20170106, 20170107])
print(multidata.loc[:,:,'LAX', ~mask]['ARR_DELAY'].mean())
#11 point
table_depflights = pd.pivot_table(data, values=['TAIL_NUM'], columns=['UNIQUE_CARRIER'], index=['DayOfWeek'], aggfunc='count')
plt.figure(2)
ax = sns.heatmap(table_depflights.corr())

#12 point
table_arr_delay = pd.pivot_table(data, values=['ARR_DELAY'], columns=['UNIQUE_CARRIER'], index=['DayOfWeek'], aggfunc='mean')
plt.figure(3)
ax2 = sns.heatmap(table_arr_delay.corr())


#13 point
table_delta_delay = pd.pivot_table(data, values=['delayDelta'], columns=['UNIQUE_CARRIER'], index=['DayOfWeek'], aggfunc='mean')
dropcolumn = [('delayDelta', 'B6'), ('delayDelta', 'VX'),
              ('delayDelta', 'EV'), ('delayDelta', 'F9'),
              ('delayDelta', 'NK'), ('delayDelta', 'OO'),
              ('delayDelta', 'WN'), ('delayDelta', 'UA')
              ]

table_delta_delay=table_delta_delay.drop(dropcolumn, axis=1)
table_delta_delay['mean'] = table_delta_delay.mean(axis=1)
print(table_delta_delay[('delayDelta', 'AA')])
plt.figure(4)
plt.plot(table_delta_delay.index, table_delta_delay[('delayDelta', 'AA')].values, label='AA')
plt.plot(table_delta_delay.index, table_delta_delay[('delayDelta', 'HA')].values, label='HA')
plt.plot(table_delta_delay.index, table_delta_delay[('delayDelta', 'DL')].values, label='DL')
plt.plot(table_delta_delay.index, table_delta_delay[('delayDelta', 'AS')].values, label='AS')
plt.plot(table_delta_delay.index, table_delta_delay['mean'], label='Mean')
plt.legend()
plt.show()
print('finito')
