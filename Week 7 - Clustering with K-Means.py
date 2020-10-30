'''
COMP809 – Data Mining and Machine Learning Lab 7 – Clustering with K-Means

This lab will cover K-Means clustering. Study the code provided below and run it in Python.
The code uses SSE (it is referred to as distortion in the code) and the cluster silhouette measure for evaluation.
The code also compares two different cluster configurations, one with K=3 and K=2. You will observe that K=3 produces a better cluster configuration.
Extend the code to use the Iris dataset. Once it is working use PCA as a pre processing step prior to applying K-Means.

'''

# KMeans module from sklean is used to perform the clustering.
# #Go to https://scikit- learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.

import numpy             as np
import matplotlib.pyplot as plt

from IPython.display  import Image
from sklearn.datasets import make_blobs
from sklearn.cluster  import KMeans
from matplotlib       import cm
from sklearn.metrics  import silhouette_samples

Figure_Path = r'/Users/chunjiangmei/Documents/OneDrive - AUT University/Semester 2 2020/COMP809_Data Mining and Machine Learning/Lab Code/'

Figure_File_1 = Figure_Path+'Week7_Output_Display the dataset.png'
Figure_File_2 = Figure_Path+'Week7_Output_Display the K-Mean Cluster.png'
Figure_File_3 = Figure_Path+'Week7_Output_Display the distortion.png'
Figure_File_4 = Figure_Path+'Week7_Output_Display the Silhouette coefficient.png'
Figure_File_5 = Figure_Path+'Week7_Output_Display the K-Mean Cluster of Bad-Clustering(using 2 clusters).png'
Figure_File_6 = Figure_Path+'Week7_Output_Display the Silhouette coefficient of Bad-Clustering(using 2 clusters) .png'

# Create a sample data set using make_blobs. This particular dataset has 2 features and 3 clusters.
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)
print(type(X))
#print(X,y)

list = [[],[],[],[]]
#print(list[0])

# Display the dataset
plt.scatter(X[:, 0],    # 取所有行的第0个数据, 即坐标X
            X[:, 1],    # 取所有行的第1个数据, 即坐标Y
            c='white',
            marker='o',
            edgecolor='black',
            s=50)
plt.grid()
plt.tight_layout()

# Customize the path to save images
plt.savefig(Figure_File_1, dpi=300)
plt.show()

# Apply k-means clustering with 3 centroids
# We set n_init=10 to run the k-means clustering algorithms 10 times independently with different random centroids to
# choose the final model as the one with the lowest SSE.
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

# Predicting cluster labels
y_km = km.fit_predict(X)
print(y_km)
print(X)
print(km.cluster_centers_)

# Visualize the clusters identified(using y_km)together with cluster labels.
plt.scatter( X[y_km == 0, 0],
             X[y_km == 0, 1],
             s=50,
             c='lightgreen',
             marker='s',
             edgecolor='black',
             label='cluster 1')

plt.scatter( X[y_km == 1, 0],
             X[y_km == 1, 1],
             s=50,
             c='orange',
             marker='o',
             edgecolor='black',
             label='cluster 2')

plt.scatter( X[y_km == 2, 0],
             X[y_km == 2, 1],
             s=50,
             c='lightblue',
             marker='v',
             edgecolor='black',
             label='cluster 3')

plt.scatter( km.cluster_centers_[:, 0],
             km.cluster_centers_[:, 1],
             s=250,
             marker='*',
             c='red',
             edgecolor='black',
             label='centroids')

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.savefig(Figure_File_2, dpi=300)
plt.show()

# Calculating Distortion
# K-均值最小化问题，是要最小化所有的数据点与其所关联的聚类中心点之间的距离之和，因此 K-均值的代价函数（又称 Distortion function , 畸变函数）
print('Distortion: %.2f' % km.inertia_)
distortions = []

# Observing the behaviour of the distortion with the number of clusters.
# Using elbow method to find optimum number of clusters
for i in range(1, 11):
     km = KMeans(n_clusters=i,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 random_state=0)
     km.fit(X)
     distortions.append(km.inertia_)
     print(distortions)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.savefig(Figure_File_3, dpi=300)
plt.show()

# Quantifying the quality of clustering via silhouette plots.
# k-means++ gives better clustering/performance than classic approach(init=’random’).
# Always recommended to use k-means++

'''
轮廓系数(silhouette coefficient) 结合了凝聚度和分离度，其计算步骤如下：

    1. 对于第 i 个对象，计算它到所属簇中所有其他对象的平均距离，记 ai （体现凝聚度）
    2. 对于第 i 个对象和不包含该对象的任意簇，计算该对象到给定簇中所有对象的平均距离，记 bi （体现分离度）
    3. 第 i 个对象的轮廓系数为 si = (bi-ai)/max(ai, bi)  从上面可以看出，轮廓系数取值为[-1, 1]，其值越大越好，
       且当值为负时，表明 ai<bi，样本被分配到错误的簇中，聚类结果不可接受。对于接近0的结果，则表明聚类结果有重叠的情况。
'''
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# Silhouette Measure/plots for cluster evaluation. Higher the score better the clustering.
cluster_labels  = np.unique(y_km)
n_clusters      = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
print(cluster_labels)
print(n_clusters)
print(silhouette_vals)
y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    print('c_silhouette_vals - Cluster Labels is : ', c)
    print(c_silhouette_vals)
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)


silhouette_avg = np.mean(silhouette_vals)

plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.savefig(Figure_File_4, dpi=300)
plt.show()

# Comparison to "bad" clustering:Same data samples are clustered using 2 cluters.
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            edgecolor='black',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='cluster 2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(Figure_File_5, dpi=300)
plt.show()


cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
     c_silhouette_vals = silhouette_vals[y_km == c]
     c_silhouette_vals.sort()
     y_ax_upper += len(c_silhouette_vals)
     color = cm.jet(float(i) / n_clusters)
     plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals,height=1.0,edgecolor='none', color=color)
     yticks.append((y_ax_lower + y_ax_upper) / 2.)
     y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.savefig(Figure_File_6, dpi=300)
plt.show()