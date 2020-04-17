import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#dataframe
data = pd.read_csv('/data-out/titanic1.csv', index_col='PassengerId')
data = data.dropna(subset=['Age'])

#replace male - 1, female - 0
data.replace({'Sex': {'male': 1, 'female': 0}}, inplace=True)

#only Pclass, sex, age fare columns
odata = data[['Pclass', 'Sex', 'Age', 'Fare']]
# another print(data.drop(columns=['Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']))

print('odata:', odata)

print('odata without nan:', odata)

clf = DecisionTreeClassifier(random_state=241)
#odata - классы, data - признаки
clf.fit(odata, data['Survived'])
importances = clf.feature_importances_
print(list(zip(odata.columns, importances)))
arg = []
for imp in reversed(importances):
    if imp > 0.3:
        imp = imp
        print(imp)
        arg.append(imp)

#importances = str( importances[2]), str( importances[3])
stri = 'Fare,Sex'

with open('/data-out/tree.txt', 'w') as f:
    f.write(stri)
    f.close()
#np.isnan(data['Age'].all())