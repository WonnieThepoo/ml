import sklearn
from sklearn import datasets
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])    # load dataset
data = newsgroups.data
targets = newsgroups.target


vectorizer = TfidfVectorizer()  # function that represent words as some view of a number
scaled = vectorizer.fit_transform(data)
#grid = {'C' : np.power(10.0, np.arange(-5, 6))}   # grid to find best C
clf = SVC(C=1, random_state=241, kernel='linear')
gener = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
#gs = GridSearchCV(clf, grid, scoring='accuracy', cv=gener, n_jobs=6)  #something that helps to check all grid
clf.fit(scaled, targets)
#x = gs.best_params_


best10 = np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]  # indexes of the heaviest words
fearture_mapping = vectorizer.get_feature_names()   # finding that best words by indexes
st = ''
setofword = []
for value in best10:
    st += str(fearture_mapping[value]) + ' '

st = ((st.split(" ")))
print(st)
st.sort()
print(st[1:])   # print that answer (only words in that order)
out = ''


"""
Should somehow to change the way to write the answer

for i in len(st)[1:12]:
    out += out + st[i] + ' '
with open('/Users/winniethepooh/PycharmProjects/ml/data-out/a_svm2.txt', 'w') as f:
    f.write(out)
    f.close()
"""
