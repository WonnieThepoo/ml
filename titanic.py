import pandas as pd
import numpy as np



data = pd.read_csv('data and out/titanic.csv', index_col='PassengerId')
passengers = len(data)

"""
Сколько м/ж
"""
male = (data['Sex'].value_counts()['male'])
female = len(data) - male
ans1 = (male, female)
with open('data and out/1.txt', 'w') as f:
    f.write(str(male))
    f.write(str(female))
    f.close()

"""
Сколько выжило в процентах
"""
survived = (data['Survived'].value_counts()[1])
ans2 = ("%.2f" % (survived/passengers*100))
with open('data and out/2.txt', 'w') as f:
    f.write(ans2)
    f.close()

"""
Сколько отн первому классу в процентах 
"""
firstclass = (data['Pclass'].value_counts()[1])
ans3 = ("%.2f" % (firstclass/passengers*100))
with open('data and out/3.txt', 'w') as f:
    f.write(ans3)
    f.close()

"""
Средний и медианный возраст
"""
ans4 = ("%.2f" % data['Age'].mean(), "%.2f" % data['Age'].median())
with open('data and out/4.txt', 'w') as f:
    f.write(str(ans4))
    f.close()

"""
Корреляция между двумя колонками
"""
ans5 = ("%.2f" % data.corr()['SibSp']['Parch'])
with open('data and out/5.txt', 'w') as f:
    f.write(str(ans5))
    f.close()


print(data['Name'])

