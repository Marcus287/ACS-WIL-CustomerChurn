!pip install yellowbrick

from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


km = KMeans(random_state=42)
elbow_vis = KElbowVisualizer(km, k=(2,10))

elbow_vis.fit(X)
print(elbow_vis.show())



kmeans = KMeans(n_clusters=4, random_state=42)
cluster_group = kmeans.fit_predict(X)

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_group, cmap='viridis', s=50, alpha=0.7)
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#            s=200, c='red', label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clustering with PCA')
plt.legend()
plt.show()