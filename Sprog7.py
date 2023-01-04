import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
df1=pd.read_csv("C:\\Users\\CSE-07\\.spyder-py3\\data8.csv")
print(df1)
f1=df1['Distance_Feature'].values
f2=df1['Speeding_Feature'].values
X=np.matrix(list(zip(f1,f2)))
plt.plot(1)
plt.subplot(511)
plt.xlim([0,100])
plt.ylim([0,50])
plt.title('Dataset')
plt.ylabel('speeding_feature')
plt.xlabel('distance_feature')
plt.scatter(f1,f2)
colors=['b','g','r']
markers=['o','v','s']

plt.plot(2)
x=plt.subplot(513)
kmeans_model=KMeans(n_clusters=3).fit(X)
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(f1[i],f2[i], color=colors[l], marker=markers[1])
plt.xlim([0,100])
plt.ylim([0,50])
plt.title('K-Means')
plt.ylabel('speeding_featur')
plt.xlabel('distance_feature')

plt.plot(3)
plt.subplot(515)
gmm=GaussianMixture(n_components=3).fit(X)
labels=gmm.predict(X)
for i, l in enumerate(labels):
    plt.plot(f1[i],f2[i],color=colors[l],marker=markers[l])
plt.xlim([0,100])
plt.ylim([0,50])
plt.title('Gaussian Mxture')
plt.ylabel('speeding_feature')
plt.xlabel('distance_feature')
plt.show()