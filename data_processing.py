import pandas as pd

df = pd.read_csv('data/r2_timings_parameters_public.csv')

locs = df['receiver_location']
c_dict = {}
for item in locs: 
    if item not in c_dict:
        c_dict[item] = 0
    c_dict[item] += 1
print(c_dict)