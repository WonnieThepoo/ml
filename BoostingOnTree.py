import matplotlib
import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/winniethepooh/PycharmProjects/ml/data-out/gbm-data.csv')

X = data.drop(columns=['Activity']).values
y = data.Activity.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

rate = [1, 0.5, 0.3, 0.2, 0.1]

def sigmoida(y_pred):
    return 1.0 / (1.0 + np.exp(-y_pred))

def plot(train_loss, test_loss):
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

"""for i in [0.2]:  # change here [0.2] (best iteration rate) to rate (array), to check and draw all graphs
    clf = RandomForestClassifier(n_estimators=36, verbose=True, random_state=241, learning_rate=i)
    clf.fit(X_train, y_train)

    scores_train = []
    for i in clf.staged_decision_function(X_train):
        scores_train.append(log_loss(y_train, [sigmoida(j) for j in i]))

    scores_test = []
    for i in clf.staged_decision_function(X_test):
        scores_test.append(log_loss(y_test, [sigmoida(j) for j in i]))
    anw = (str(min(scores_test))[0:4] + ' ' + str(scores_test.index(min(scores_test)))) # answer for 4
    with open('/Users/winniethepooh/PycharmProjects/ml/data-out/BoostingOnTree.txt', 'w') as f:
        f.write(str(anw))
        f.close()
    plot(scores_train, scores_test)"""

# 5
clf = RandomForestClassifier(n_estimators=36, random_state=241)
clf.fit(X_train, y_train)

score_train = log_loss(y_train, clf.predict_proba(X_train))
score_test = log_loss(y_test, clf.predict_proba(X_test))
print(score_test)
with open('/Users/winniethepooh/PycharmProjects/ml/data-out/BoostingOnTree.txt', 'w') as f:
    f.write(str(score_test)[0:4])
    f.close()

