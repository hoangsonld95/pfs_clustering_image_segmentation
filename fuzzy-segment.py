import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def fcm(data, n_clusters=1, n_init=30, m=2, max_iter=300, tol=1e-16):

    min_cost = np.inf
    for iter_init in range(n_init):

        # Randomly initialize centers
        centers = data[np.random.choice(
            data.shape[0], size=n_clusters, replace=False
            ), :]

        # Compute initial distances
        # Zeros are replaced by eps to avoid division issues
        dist = np.fmax(
            cdist(centers, data, metric='sqeuclidean'),
            np.finfo(np.float64).eps
        )

        for iter1 in range(max_iter):

            # Compute memberships       
            u = (1 / dist) ** (1 / (m-1))
            um = (u / u.sum(axis=0))**m

            # Recompute centers
            prev_centers = centers
            centers = um.dot(data) / um.sum(axis=1)[:, None]

            dist = cdist(centers, data, metric='sqeuclidean')

            if np.linalg.norm(centers - prev_centers) < tol:
                break

        # Compute cost
        cost = np.sum(um * dist)
        if cost < min_cost:
            min_cost = cost
            min_centers = centers
            mem = um.argmax(axis=0)

    return min_centers, mem


if __name__ == '__main__':
    data = np.loadtxt('grid5_edit.txt')
    labels = data[:, -1]
    k = len(np.unique(labels))
    data = data[:, 0:-1]

    repeats = 10

    # Time This
    fcm_time = 0
    for iter1 in range(repeats):
        fcm_start = time.time()
        centers, mem = fcm(
            data, n_clusters=k, n_init=30, m=2, max_iter=300, tol=1e-16
        )
        fcm_time += (time.time() - fcm_start)
    print('Average FCM time =', fcm_time/repeats)

    # Time This (as well)
    km_time = 0
    for iter1 in range(repeats):
        km_start = time.time()
        km1 = KMeans(
            n_clusters=k, n_init=30, max_iter=300, tol=1e-16
        ).fit(data)
        km_time += (time.time() - km_start)
    print('Average kMeans time =', km_time/repeats)

    print('Ratio of time =', fcm_time / km_time)