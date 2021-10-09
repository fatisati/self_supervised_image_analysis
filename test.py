import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1,2,3]})
idx = df['a'] == 1
a = [2,3,4]

print(~idx)
print(np.array(a)[~idx])

