import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection, sklearn.neighbors, sklearn.preprocessing, sklearn.datasets

data = sklearn.datasets.load_boston()

targets = data['target']
scale_data = (sklearn.preprocessing.scale(X=data['data']))


values = np.linspace(1, 10, num=200)

acc = []
for value in values:
    classfile = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=value)
    gener = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    accuracy =  np.mean(sklearn.model_selection.cross_val_score(classfile, cv=gener, X=scale_data, y=targets,
                                                              scoring='neg_mean_squared_error'))
    acc.append(accuracy)


print('max with scaling:', max(acc), acc.index(max(acc))+1)
with open('data and out/metric.txt', 'w') as f:
    f.write(str(max(acc)))   #  max accuracy
    f.close()