import sklearn.metrics as mtr
import numpy as np
import pandas as pd
import math

"""
Логическая регрессия
Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная 
на которой принимает значения -1 или 1.
Убедитесь, что выше выписаны правильные формулы для градиентного спуска. Обратите внимание, 
что мы используем полноценный градиентный спуск, а не его стохастический вариант!
Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10) 
логистической регрессии. 
Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов на соседних итерациях 
должно быть не больше 1e-5). 
Рекомендуется ограничить сверху число итераций десятью тысячами.
Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании? Эти величины будут ответом на 
задание. 
В качестве ответа приведите два числа через пробел. Обратите внимание, что на вход функции 
roc_auc_score нужно подавать оценки вероятностей, подсчитанные обученным алгоритмом. Для этого воспользуйтесь 
сигмоидной функцией: a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).
"""


# funct to find Euclidean distance
def eucd(v, u):
   return np.sqrt(np.sum((v - u) ** 2))

# sigmoid function
def sigmoid(w1, w2, x1, x2):
    return 1/(1+np.exp(-w1*x1 - w2*x2))

data = np.genfromtxt('/Users/winniethepooh/PycharmProjects/ml/data-out/data-logistic.csv', delimiter=',')
y = data[:, 0]
x1 = data[:, 1]
x2 = data[:, 2]

C = 0   # zero for L1 reg
w1 = 0
w2 = 0
eukd = 1
L = len(y)
dict = [0.1]    # add more variables for different k
it = 0
for k in dict:
    while eukd > 10 ** (-5) and it < 10000:
        sum1 = 0
        sum2 = 0
        for i in range(len(y)):
            a = (1 - 1 / (1 + np.exp(-y[i] * (w1 * x1[i] + w2 * x2[i]))))
            sum1 += y[i] * x1[i] * a
            sum2 += y[i] * x2[i] * a
        w1 += k * 1 / L * sum1 + k * C * w1
        w2 += k * 1 / L * sum2 + k * C * w2
        it += 1
        eukd = eucd(w1, w2)

    ans = mtr.roc_auc_score(y_true=y, y_score=sigmoid(w1, w2, x1, x2))

    print('L1:', ans)

    C = 10  # 10 for L2 reg
    w1 = 0
    w2 = 0
    eukd = 1
    it = 0

    while eukd > 10 ** (-5) and it < 10000:
        sum1 = 0
        sum2 = 0
        for i in range(len(y)):
            a = (1 - 1 / (1 + np.exp(-y[i] * (w1 * x1[i] + w2 * x2[i]))))
            sum1 += y[i] * x1[i] * a
            sum2 += y[i] * x2[i] * a
        w1 += k * 1 / L * sum1 - k * C * w1
        w2 += k * 1 / L * sum2 - k * C * w2
        it += 1
        eukd = eucd(w1, w2)

    answ = (str(ans))
    ans = mtr.roc_auc_score(y_true=y, y_score=sigmoid(w1, w2, x1, x2))
    print('L2:', ans)
    answ += str(ans)

with open('/Users/winniethepooh/PycharmProjects/ml/data-out/logicRegression.txt', 'w') as f:
    f.write(answ)  # max accuracy
    f.close()