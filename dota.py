import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('/Users/winniethepooh/PycharmProjects/ml/data-out/dota.csv')
test = pd.read_csv('/Users/winniethepooh/PycharmProjects/ml/data-out/dota_test.csv')


def bestEst(clf, gener, X, y):
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=gener, n_jobs=6)
    gs.fit(X, y)
    x = gs.best_estimator_
    print(x)


def RFclf():
    counter = 0
    print('Колонки с пропущенными значениями:')
    for i in data.count():
        if (i != 97230):
            x = data.iloc[:, counter]
            print('Name:', x.name, '\nВсего пропущенных значений:',
                  97230 - x.count(), 'из 97230',
                  'или:', 100 - round(x.count() / 97230 * 100), '%')
        counter += 1
    # Объяснение - все эти покупки предметов или действия могли не совершиться в первые 5 минут игры
    # (в силу особенностей игрового процесса), при этом покупка курьера совершается почти в 100% за первые 5 минут игры
    data0 = data.fillna(0)
    dataMean = data.apply(lambda x: x.fillna(x.mean()), axis=0)
    dataForTree = data.fillna(1000000)
    X_0 = data0.drop(columns=['duration',
                              'radiant_win',
                              'tower_status_radiant',
                              'tower_status_dire',
                              'barracks_status_radiant',
                              'barracks_status_dire'])
    y_0 = data0['radiant_win']  # целевая переменная
    X_mean = dataMean.drop(columns=['duration',
                                    'radiant_win',
                                    'tower_status_radiant',
                                    'tower_status_dire',
                                    'barracks_status_radiant',
                                    'barracks_status_dire'])
    y_mean = dataMean['radiant_win']  # целевая переменная
    X_tree = dataForTree.drop(columns=['duration',
                                       'radiant_win',
                                       'tower_status_radiant',
                                       'tower_status_dire',
                                       'barracks_status_radiant',
                                       'barracks_status_dire'])
    y_tree = dataForTree['radiant_win']  # целевая переменная
    X_datas = [X_0, X_mean, X_tree]
    y_datas = [y_0, y_mean, y_tree]


    for x in [10, 20, 30]:
        clf = RandomForestClassifier(n_estimators=x, random_state=241)
        gener = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, j in zip(X_datas, y_datas):
            start = time.time()
            clf.fit(i, j)
            score = cross_val_score(clf, i, j, cv=gener, scoring='roc_auc')
            print('Numbers of tree:', x, 'time: %.2f' % float(time.time() - start), 'score: %.4f' % float(score.mean()))
    # Время затрачнное на кросс-валидацию при 30 деревьях ~90 секунд
    # Действительно, точность выросла при большем количестве базовых деревьев, а при использовании датасета, подготовленного
    # для деревьев (замена пропусков на очень большие значения), максимально удалось достичь среднего значения в 0.671


def linRegr():
    #1
    scaler = StandardScaler()
    X = data.drop(columns=['duration',
                           'radiant_win',
                           'tower_status_radiant',
                           'tower_status_dire',
                           'barracks_status_radiant',
                           'barracks_status_dire'])
    y = data['radiant_win']  # целевая переменная
    X = scaler.fit_transform(X.fillna(0))
    start = time.time()
    clf = LogisticRegression(C=10, random_state=241)
    gener = KFold(n_splits=5, shuffle=True, random_state=42)
    #bestEst(clf, gener, X,y)
    clf.fit(X, y)
    score1 = np.mean(cross_val_score(clf, X, y, cv=gener, scoring='roc_auc'))
    print('Оценка кросс-валидации с признаками-героями: %.4f' % float(score1), 'Время: %.2f' % float(time.time() - start))
    # Такого же значения (>0.717) можно добиться и с использованием
    # град. бустинга над деревьями, но на это уйдет гораздо больше времени


    #2
    scaler = StandardScaler()
    X = data.drop(columns=['lobby_type',
                           'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                           'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero',
                           'duration',
                           'radiant_win',
                           'tower_status_radiant',
                           'tower_status_dire',
                           'barracks_status_radiant',
                           'barracks_status_dire'])

    X = scaler.fit_transform(X.fillna(0))
    start = time.time()
    clf = LogisticRegression(C=0.1, random_state=241)
    gener = KFold(n_splits=5, shuffle=True, random_state=42)
    #bestEst(clf, gener, X,y)
    clf.fit(X, y)
    score2 = np.mean(cross_val_score(clf, X, y, cv=gener, scoring='roc_auc'))
    print('Оценка кросс-валидации без признаков-героев: %.8f' % float(score2), 'Время: %.2f' % float(time.time() - start))
    print('Разница оценок: %.8f' % float(score2 - score1))


    #3
    heroes = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
              'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
    arr = []
    for i in heroes:
        x = int(max(data[i].unique()))
        arr.append(x)
    print('Всего героев:', max(arr))    # 112 героев


    #4
    X = data.drop(columns=['lobby_type',
                           'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                           'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero',
                           'duration',
                           'radiant_win',
                           'tower_status_radiant',
                           'tower_status_dire',
                           'barracks_status_radiant',
                           'barracks_status_dire'])
    X_pick = np.zeros((data.shape[0], 112))
    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.loc[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, data.loc[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    X_pick = pd.DataFrame(X_pick)
    X = pd.concat([X, X_pick], axis=1, sort=False)


    #5
    scaler = StandardScaler()
    X = scaler.fit_transform(X.fillna(0))
    start = time.time()
    clf = LogisticRegression(C=0.001, random_state=241)
    gener = KFold(n_splits=5, shuffle=True, random_state=42)
    #bestEst(clf, gener, X,y)
    clf.fit(X, y)
    score3 = np.mean(cross_val_score(clf, X, y, cv=gener, scoring='roc_auc'))
    print('Оценка кросс-валидации с мешком героев:', score3, 'Время: %.4f' % float(time.time() - start))
    print('Разница оценок 2 и 3: %.4f' % float(score3 - score2))
    # Добавление мешка героев улучшило показания на кросс-валидации до 0.75, это можно объяснить тем, что
    # разряженная матрица героев помогает построить верное предсказание


    #6
    X_test = test.drop(columns=['lobby_type',
                                'r1_hero','r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                                'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])
    X_pick = np.zeros((data.shape[0], 112))
    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.loc[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, data.loc[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    X_pick = pd.DataFrame(X_pick)
    X_test = pd.concat([X_test, X_pick], axis=1, sort=False)
    X_test = scaler.transform(X_test.fillna(0))


    anw = clf.predict_proba(X_test)[:, 1]
    print('Оценка  средняя: %.4f' % float(np.mean(anw)))  # 0.65 на тестовой выборке
    print('Оценка минимальная: %.4f' % float(min(anw)))
    print('Оценка  максимальная: %.4f' % float(max(anw)))


if __name__ == '__main__':
    RFclf()
    linRegr()