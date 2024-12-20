import pandas as pd
import haversine as hs

df = pd.read_csv('data/r1_timings_parameters_public.csv')

loc_df = pd.read_csv('countries.csv')
loc_dict = {}
for i, item in loc_df.iterrows():
    print
    loc_dict[item['ISO']] = (item['latitude'],item['longitude'] )
#locs = df['receiver_cc']
distances = []
s_countries= {}
r_countries= {}
for i, item in df.iterrows():
    sender_loc = loc_dict[item['sender_cc'].upper()]
    receiver_loc = loc_dict[item['receiver_cc'].upper()]
    dist = hs.haversine(sender_loc, receiver_loc)
    distances.append(int(dist))
    r_countries[item['receiver_cc']] = 0
    s_countries[item['sender_cc']] = 0
print(r_countries)
print(s_countries)


df['SR_Distance'] = distances
df = df.rename(columns={"Unnamed: 0": 'id'})
#df.to_csv('data/r1_timings_distance.csv', index=False)