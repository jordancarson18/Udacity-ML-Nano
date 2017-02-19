import pandas as pd
import numpy as np
import sys

print('Python version ' + sys.version)
print('Pandas version: ' + pd.__version__)


d = {'name': pd.Series(['Jord','Ccarr','Louis'], index=['a','b','c']),'age':pd.Series([18,22,25],index=['a','b','c']),'yay':pd.Series(['yay','nay','NULL'],index=['a','b','c'])}
df = pd.DataFrame(d)

df['New Column'] = [[1,2,3],[3,4,5],[6,7,8]]


ds = df.stack()
du = ds.unstack()
dt = df.T

print(dt)

print(du)
print(ds)
print(df)
print(df.dtypes)
