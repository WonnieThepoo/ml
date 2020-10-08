import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA

data = pd.read_csv('/data-out/djia_index.csv')
data_close = pd.read_csv('/data-out/close_prices.csv')


data_wd = data['^DJI']
data_close_wd = data_close.drop(labels='date', axis=1)
x = np.array(data_close_wd)
clf = PCA(n_components=10)
clf.fit(x)


disp = 0
cnt = 0
for cmp in clf.explained_variance_ratio_:
    disp += cmp
    cnt += 1
    if disp > 0.9:
        break
print('min of components: ', cnt)
with open('/data-out/index.txt', 'w') as f:
    f.write(str(cnt))
    f.close()


trans = pd.DataFrame(clf.transform(x))
corr = trans[0].corr(data_wd)
print('Corr: %0.2f' % corr)
with open('/data-out/index.txt', 'w') as f:
    f.write(str('%0.2f' % corr))
    f.close()

idx = np.argmax(clf.components_[0]) + 1
data_close = pd.read_csv('/data-out/close_prices.csv')