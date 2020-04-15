import numpy as np
import pandas as pd
"""x = np.random.normal(loc=1, scale=10, size=(1000, 50)) random matrix
print(x)
print('/n')
m = np.mean(x, axis=0) среднее знач по стобцам
std = np.std(x, axis=0) стд отколноение
X_norm = (x-m)/std 
print(X_norm)

Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])

f = np.sum(Z, axis=1) сумма по строкам
print(np.nonzero(f > 10)) 

a = np.eye(3)
b = np.eye(3)
ab = np.vstack((a, b)) объединение матриц
print(ab)
"""