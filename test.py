import pandas as pd

df = pd.DataFrame([{'a':1}])
df2 = pd.DataFrame([{'a': 2}])
df.append(df2)
print(df)

