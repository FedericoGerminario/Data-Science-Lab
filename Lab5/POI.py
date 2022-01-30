import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def get_id_cell(lat, lon, minlat, maxlat, minlon, maxlon, grids):
    unit_lat = (maxlat - minlat)/grids
    unit_lon = (maxlon - minlon)/grids
    idlat = round((lat - minlat)/unit_lat + 0.5)
    idlon = round((lon - minlon)/unit_lon + 0.5) + grids
    return idlat*idlon


def plot_in_map(image, df_amenity, df_shop, df_public_transport, df_highway):
    img = plt.imread(image)  
    fig, ax = plt.subplots()
    ax.scatter(df_amenity['@lon'], df_amenity['@lat'], c='r', s=10, marker='.')
    ax.scatter(df_shop['@lon'], df_shop['@lat'], c='b', s=10, marker='.')
    ax.scatter(df_public_transport['@lon'], df_public_transport['@lat'], c='y', s=10, marker='.')
    ax.scatter(df_highway['@lon'], df_highway['@lat'], c='g', s=10, marker='.')
    ax.set_xlim(min(df_amenity['@lon']), max(df_amenity['@lon']))
    ax.set_ylim(min(df_amenity['@lat']) -0.01, max(df_amenity['@lat']))
    ax.imshow(img, extent=(min(df_amenity['@lon']), max(df_amenity['@lon']) ,min(df_amenity['@lat']) -0.01, max(df_amenity['@lat'])))
    print(ax.grid())


nypois = pd.read_csv('ny_municipality_pois_id.csv', header=None)
dataset = pd.read_csv('pois_all_info.txt', sep='\t', low_memory=False)

dataset.index = dataset['@id']
df = dataset.loc[nypois[0]]

categories = ['amenity', 'shop', 'public_transport', 'highway']

#plot_in_map('New_York_City_Map.PNG')


masks = {}

for el in categories:
    to_plot = df.set_index([el, "@type"]).count(level=el)['@id']
    to_plot = to_plot/(to_plot.sum()) * 100
    mask = (to_plot > 1)
    to_plot = to_plot.loc[mask]
    mask2 = [el for el in mask.index if mask.loc[el] == True]
    masks[el] = df[el].isin(mask2)
    to_plot.plot(kind='bar')


df_amenity = df.loc[masks['amenity']]
df_shop = df.loc[masks['shop']]
df_public_transport = df.loc[masks['public_transport']]
df_highway = df.loc[masks['highway']]

frequent_items = pd.concat((df_amenity, df_shop, df_public_transport, df_highway))
#plot_in_map('New_York_City_Map.PNG', df_amenity, df_shop, df_public_transport, df_highway)


frequent_items['grid_id'] =get_id_cell(frequent_items['@lat'], frequent_items['@lon'], min(frequent_items['@lat']),
                           max(frequent_items['@lat']), min(frequent_items['@lon']), max(frequent_items['@lon']), 3)

#data.index = set(frequent_items['grid_id'])
#print(data.index)

column_amenity = set(df_amenity['amenity'])


#6 point
#frequent_items['grid_id'] = frequent_items['grid_id'].astype(int)
frequent_amenity = frequent_items.drop(['@lat', '@lon', '@type', 'name', 'shop', 'public_transport', 'highway'], axis=1)
frequent_shop = frequent_items.drop(['@lat', '@lon', '@type', 'name', 'amenity', 'public_transport', 'highway'], axis=1)
frequent_public_transport = frequent_items.drop(['@lat', '@lon', '@type', 'name', 'amenity', 'shop', 'highway'], axis=1)
frequent_highway = frequent_items.drop(['@lat', '@lon', '@type', 'name', 'amenity', 'public_transport', 'shop'], axis=1)



frequent_amenity = pd.pivot_table(frequent_amenity, index='grid_id', columns='amenity', aggfunc='count')
frequent_shop = pd.pivot_table(frequent_shop, index='grid_id', columns='shop', aggfunc='count')
frequent_public_transport = pd.pivot_table(frequent_public_transport, index='grid_id', columns='public_transport', aggfunc='count')
frequent_highway = pd.pivot_table(frequent_highway, index='grid_id', columns='highway', aggfunc='count')


frequent_corr = frequent_amenity.append(frequent_shop).groupby('grid_id').sum()

#7 point
correlation = frequent_corr.corr()
plt.figure(2)
sns.heatmap(correlation)
plt.show()


#frequent_items = frequent_items.groupby(['grid_id']).count()
#print(frequent_items)

#table = pd.pivot_table(frequent_items, index=['grid_id'], columns=['amenity'], aggfunc='count')
#print(table)
print('finito')
