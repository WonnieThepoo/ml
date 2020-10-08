import numpy as np
import sklearn
import pandas as pd
from sklearn.cluster import KMeans
from skimage.io import imread
import pylab
import math
import matplotlib.pyplot as plot
from skimage import img_as_float as iaf

image = imread('/Users/winniethepooh/PycharmProjects/ml/data-out/_3160f0832cf89866f4cc20e07ddf1a67_parrots.jpg')
image = iaf(image)



r = image[:, :, 0].ravel()
g = image[:, :, 1].ravel()
b = image[:, :, 2].ravel()
rgb = np.transpose(np.vstack((r, g, b)))

clf = KMeans(random_state=241, init='k-means++')
clf.fit(rgb)
clusters = clf.labels_
avg = clf.cluster_centers_#
cls_img = np.reshape(clusters, (-1, 713))

mean_img = np.copy(image)
for cluster in range(0, clf.n_clusters):
    mean_r = np.mean(mean_img[:, :, 0][cls_img == cluster])
    mean_g = np.mean(mean_img[:, :, 1][cls_img == cluster])
    mean_b = np.mean(mean_img[:, :, 2][cls_img == cluster])
    mean_img[cls_img == cluster] = avg[cluster]
    plot.imshow(mean_img)

med_img = np.copy(image)
for cluster in range(0, clf.n_clusters):
    median_r = np.median(med_img[:, :, 0][cls_img == cluster])
    median_g = np.median(med_img[:, :, 1][cls_img == cluster])
    median_b = np.median(med_img[:, :, 2][cls_img == cluster])
    med_img[cls_img == cluster] = avg[cluster]
    plot.imshow(med_img)

def PSNR(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    psnr = 10 * math.log10(np.max(image1) / mse)
    return psnr

psnr1 = PSNR(image, med_img)
psnr2 = PSNR(image, mean_img)
print(psnr1, psnr2)

for i in range(1, 21):
    clf = KMeans(random_state=241, init='k-means++', n_clusters=i)
    clf.fit(rgb)
    clusters = clf.labels_
    avg = clf.cluster_centers_
    cls_img = np.reshape(clusters, (-1, 713))
    img = np.copy(image)
    for cluster in range(0, i):
        img[cls_img == cluster] = avg[cluster]
    print(i, PSNR(image, img))