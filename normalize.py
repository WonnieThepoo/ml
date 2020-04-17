import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection, sklearn.linear_model, sklearn.preprocessing, sklearn.metrics
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train = np.genfromtxt('/Users/winniethepooh/PycharmProjects/ml/data-out/perceptron-train.csv', delimiter=',')
test = np.genfromtxt('/Users/winniethepooh/PycharmProjects/ml/data-out/perceptron-test.csv', delimiter=',')
#data: targets and signs
train_y = train[:, 0]
train_x = train[:, 1:]
test_y = test[:, 0]
test_x = test[:, 1:]

#training Perceptron
clf = Perceptron(random_state=241, max_iter=5, tol=None) #generator
clf.fit(train_x, train_y) #training
predictions = clf.predict(test_x)
acc = accuracy_score(test_y, predictions) #accuracy of predictions (diff between pred of test and
print('accuracy without scaling', acc)

#scaling
scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)
scaled_test_x = scaler.transform(test_x)


clf.fit(X=scaled_train_x, y=train_y)    # training Perceptron with scaling signs
predictions_test = clf.predict(scaled_test_x)

scale_acc = accuracy_score(test_y, predictions_test)    # accuracy with scaling
print('accuracy with scaling:', scale_acc)

diff = round((scale_acc - acc), 3)
print('difference:', diff)
with open('/Users/winniethepooh/PycharmProjects/ml/data-out/normalize.txt', 'w') as f:
    f.write(str(diff))
    f.close()