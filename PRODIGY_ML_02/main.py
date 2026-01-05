import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

X = data.iloc[:, [3, 4]].values

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure()

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1])
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1])
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1])
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1])
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()
