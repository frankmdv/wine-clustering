from models import Metrics
from read import read_wine
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


print("Loading data...")
data, d_classes = read_wine()

print("Analyzing data")

f_classes = np.unique(d_classes)
cols = len(data.iloc[0, :])
average = np.zeros([len(f_classes), cols])

iters = 100

metrics = [Metrics(), Metrics(), Metrics()]

f_data = np.zeros([data.shape[0], cols + 1])
f_data[:, 0:cols] = data
f_data[:, cols] = d_classes.values

for i in range(iters):
    print(str(i) + " / " + str(iters))

    training_data, testing_data = train_test_split(f_data, test_size=0.3)

    df_train = training_data[:, 0:cols]
    df_val = testing_data[:, 0:cols]

    real_train_class = training_data[:, cols]
    real_val_class = testing_data[:, cols]

    real_average = np.zeros([len(f_classes), cols])
    for j in range(len(f_classes)):
        indexes = np.where(real_train_class == j)
        real_average[j, :] = np.mean(df_train[indexes, :], axis=1)

    # Modelo k-means --------------------------------------------------------

    kmeans = KMeans(n_clusters=3, n_init=25)
    kmeans.fit(df_train)

    pred_train_class = kmeans.predict(X=df_train)
    pred_val_class = kmeans.predict(X=df_val)

    pred_average = kmeans.cluster_centers_

    rename = np.zeros([len(f_classes), 1])
    for j in range(len(f_classes)):
        rename[j] = -1  # valor por defecto

    euclidean_dist = euclidean_distances(real_average, pred_average)

    for j in range(len(f_classes)):
        t = np.argmin(euclidean_dist[j, :])
        rename[j] = t
        euclidean_dist[:, t] = 10

    pred_val_class2 = np.zeros(pred_val_class.shape)
    for j in range(len(pred_val_class)):
        pred_val_class2[j] = np.where(rename == pred_val_class[j])[0]

    # se calcula la matriz de confusion
    cm = confusion_matrix(real_val_class, pred_val_class2)

    Metrics.set_confusion_matrix(cm)
    for m in range(3):
        Metrics.set_values(m)
        metrics[m].set_metrics()


index_values = ["Clase 1", "Clase 2", "Clase 3"]
column_values = ["Accuracy %", "Precision %", "Sensitivity %", "F1 %"]

res_means = np.zeros([3, 4])
for i in range(res_means.shape[0]):
    res_means[i, 0] = np.round(np.mean(metrics[i].accuracy), 2)
    res_means[i, 1] = np.round(np.mean(metrics[i].precision), 2)
    res_means[i, 2] = np.round(np.mean(metrics[i].sensitivity), 2)
    res_means[i, 3] = np.round(np.mean(metrics[i].f1), 2)

res_means = pd.DataFrame(data=res_means, index=index_values, columns=column_values)

print(" ", "K-Means", res_means, sep="\n")
