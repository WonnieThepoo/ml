import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection, sklearn.neighbors, sklearn.preprocessing

data = np.genfromtxt('wine.data', delimiter=',')
# classes:
y = data[:, 0]
# sings:
x = data[:, 1:]

acc = []
for k in range(1, 51):
    classfile = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    gener = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    accuracy = np.mean(sklearn.model_selection.cross_val_score(classfile, cv=gener, X=x, y=y, scoring='accuracy'))
    acc.append(accuracy)

print('max without scaling :', max(acc), acc.index(max(acc)))

with open('data and out/kNN1.txt', 'w') as f:
    f.write(str(acc.index(max(acc))))  # index of max accuracy
    f.close()

with open('data and out/kNN2.txt', 'w') as f:
    f.write(str(max(acc)))  # max accuracy
    f.close()

acc = []
for k in range(1, 51):
    classfile = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    gener = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)

    #to similar scale, that changes k from 1 to 28
    x = (sklearn.preprocessing.scale(X=x))

    accuracy = np.mean(sklearn.model_selection.cross_val_score(classfile, cv=gener, X=x, y=y, scoring='accuracy'))
    acc.append(accuracy)

print('max with scaling:', max(acc), acc.index(max(acc)))

with open('data and out/kNN3.txt', 'w') as f:
    f.write(str(acc.index(max(acc))))   # index of max accuracy
    f.close()

with open('data and out/kNN4.txt', 'w') as f:
    f.write(str(max(acc)))  # max accuracy
    f.close()
