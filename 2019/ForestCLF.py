import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection

data = pd.read_csv('/data-out/abalone.csv')

data.replace({'Sex': {'M': 1, 'F': -1, 'I': 0}}, inplace=True)
y = data['Rings']
X = data.drop(columns=['Rings'])

anw = ''
for k in range(1, 51):
    clf = RandomForestRegressor(random_state=1, n_estimators=k)
    gener = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    clf.fit(X, y)
    accuracy = np.mean(sklearn.model_selection.cross_val_score(clf, cv=gener, X=X, y=y, scoring='r2'))
    if round(accuracy, 2) > 0.52:
        print(accuracy, k)
        anw = str(k)
        break

with open('/data-out/RandomForestCLF.txt', 'w') as f:
    f.write(str(anw))
    f.close()
