import pandas as pd
import numpy as np
import scipy.sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge


"""
Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv 
(либо его заархивированную версию salary-train.zip).
Проведите предобработку:
Приведите тексты к нижнему регистру (text.lower()).
Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. 
Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). 
Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:

Примените TfidfVectorizer для преобразования текстов в векторы признаков. Оставьте только те слова, 
которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. 
Код для этого был приведен выше.
Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, что матрицы для текстов и 
категориальных признаков являются разреженными. Для объединения их столбцов нужно 
воспользоваться функцией scipy.sparse.hstack.
3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241.
Целевая переменная записана в столбце SalaryNormalized.
4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. 
Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
"""


data = pd.read_csv('/Users/winniethepooh/PycharmProjects/ml/data-out/salary-train.csv')
data_test = pd.read_csv('/Users/winniethepooh/PycharmProjects/ml/data-out/salary-test-mini.csv')

data['FullDescription'] = data['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)

y_train = data['SalaryNormalized']
x_train = data[['FullDescription', 'LocationNormalized', 'ContractTime']]
y_test = data_test['SalaryNormalized']
x_test = data_test[['FullDescription', 'LocationNormalized', 'ContractTime']]

vectorizer = TfidfVectorizer(min_df=5)  # function that represent words as some view of a number
scaled = vectorizer.fit_transform(data['FullDescription'])
test_scaled = vectorizer.transform(data_test['FullDescription'])


enc = DictVectorizer()
X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

train_m = scipy.sparse.hstack((scaled, X_train_categ))
test_m = scipy.sparse.hstack((test_scaled, X_test_categ))

clf = Ridge(random_state=241, alpha=1)
clf.fit(train_m, y_train)
anw = clf.predict(test_m)
print(anw)
anw = str(str(anw[1])[1:9] + ' ' + str(anw[2])[1:9])
with open('/Users/winniethepooh/PycharmProjects/ml/data-out/LinearRegression.txt', 'w') as f:
    f.write(str(max(anw)))   # max accuracy
    f.close()
