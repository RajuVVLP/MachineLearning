

import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir('D:/Data Science/Data/')

train = pd.read_csv('house_train.csv')
#train.info()

plt.scatter(x = train['Neighborhood'], y = train['SalePrice'],data = train)

plt.xticks(rotation=70)
plt.show()