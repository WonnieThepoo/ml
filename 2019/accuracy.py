import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve


data = np.genfromtxt('/Users/winniethepooh/PycharmProjects/ml/data-out/classification.csv', delimiter=',', skip_header=1)
true = data[:, 0]
pred = data[:, 1]

# error matrix
tp = 0 #true positive
fn = 0 #false negative
fp = 0 #false positive
tn = 0 #true negative
for i, j in zip(true, pred):
    if i == 1 and j == 1:
        tp += 1
    elif i == 1 and j == 0:
        fn += 1
    elif i == 0 and j == 1:
        fp += 1
    else:
        tn += 1
st = str(tp) + ' ' + str(fp) + ' ' + str(fn) + ' ' + str(tn) #first answer

#accury score of varios metrics
acc_s = accuracy_score(true, pred)
prec_s = precision_score(true, pred)
rec_s = recall_score(true, pred)
f1_s = f1_score(true, pred)
st = str(round(acc_s, 2)) + ' ' + str(round(prec_s, 2)) + ' ' + str(round(rec_s, 2)) + ' ' + str(round(f1_s, 2)) #second answer
with open('/data-out/tableOfAccuracy', 'w') as f:
    f.write(st)
    f.close()


data = np.genfromtxt('/Users/winniethepooh/PycharmProjects/ml/data-out/scores.csv', delimiter=',', skip_header=1)
true = data[:, 0]
score_logreg = data[:, 1]
score_svm = data[:, 2]
score_knn = data[:, 3]
score_tree = data[:, 4]
logreg = roc_auc_score(true, score_logreg)
svm = roc_auc_score(true, score_svm)
knn = roc_auc_score(true, score_knn)
tree = roc_auc_score(true, score_tree)
st = str(round(logreg, 2)) + ' ' + str(round(svm, 2)) + ' ' + str(round(knn, 2)) + ' ' + str(round(tree, 2)) #third answer
with open('/data-out/tableOfAccuracy', 'w') as f:
    f.write(st)
    f.close()


ans = []
precisions = []
logreg = precision_recall_curve(true, score_logreg)
for x, y, z in zip(logreg[0], logreg[1], logreg[2]):
    if y > 0.7:
        precisions.append(x)
ans.append(max(precisions))
precisions = []
svm = precision_recall_curve(true, score_svm)
for x, y, z in zip(svm[0], svm[1], svm[2]):
    if y > 0.7:
        precisions.append(x)

ans.append(max(precisions))
precisions = []
knn = precision_recall_curve(true, score_knn)
for x, y, z in zip(knn[0], knn[1], knn[2]):
    if y > 0.7:
        precisions.append(x)
ans.append(max(precisions))
precisions = []
tree = precision_recall_curve(true, score_tree)
for x, y, z in zip(tree[0], tree[1], tree[2]):
    if y > 0.7:
        precisions.append(x)
ans.append(max(precisions))

print(ans) #last answer is here
ans = str(max(ans))
print(ans)


