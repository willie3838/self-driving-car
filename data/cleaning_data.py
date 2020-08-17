import pandas as pd
import numpy as np

df = pd.read_pickle('data.pkl')
df.drop(list(df.filter(regex = 'Unnamed')), axis = 1, inplace = True)
df.drop(columns=['Action1','Action2','Level',"Observations2"], inplace=True)
#df.drop_duplicates(inplace=True)
print(df)

'''df.drop(list(df.filter(regex = 'Unnamed')), axis = 1, inplace = True)
#df.drop_duplicates(inplace=True)

df.drop(columns=['Action1','Action2',"Observations","Critical"], inplace=True)
print(df)'''

df.to_pickle('critical.pkl')