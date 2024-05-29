from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.spatial.distance import cityblock


@dataclass
class ImageKMeans:
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

    def fit(self, image: np.ndarray) -> ImageKMeans:
        """Fit k-means to the image"""

        X = self._image_as_array(image)
        self._init_centroids(X)

        # iterate until reaching max_iter
        for _ in range(self.max_iter):
            y = self._assign_centroids(X)
            self._update_centroids(X, y)

        return self

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return the labels of the image"""
        X = self._image_as_array(image)

        # assign each sample to the closest centroid
        labels = self._assign_centroids(X)
        return labels.reshape(image.shape[0], image.shape[1])

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return the compressed image"""
        clusters = self.predict(image)
        image_comp = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image_comp[i, j] = self.centroids_[int(clusters[i, j])]

        return image_comp

    def _update_centroids(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the centroids by taking the mean of its samples"""
        new_centers = [0] * self.n_clusters
        for i in range(self.n_clusters):
            new_centers[i] = np.median(X[y == i], axis=0).astype(int)
        self.centroids_ = np.array(new_centers)

    def _image_as_array(self, image: np.ndarray) -> np.ndarray:
        """Convert image to pixel array"""
        X = image.reshape(-1, 3)
        return X

    def _init_centroids(self, X: np.ndarray) -> None:
        if X.min() < 0 or X.max() > 255:
            raise ValueError()
        if isinstance(self.init, str) and self.init != "random":
            raise ValueError()
        if isinstance(self.init, np.ndarray) and (
                self.init.shape != (
                self.n_clusters, 3) or self.init.min() < 0 or self.init.max() > 255 or not np.array_equal(
            np.unique(self.init, axis=0), self.init)):
            raise ValueError()
        if isinstance(self.init, np.ndarray):
            self.centroids_ = self.init
            return
        if self.init == "random":
            np.random.seed(self.random_state)
            random_index = np.random.choice(X.shape[0], self.n_clusters)
            self.centroids_ = X[random_index]
            return
        raise TypeError()

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid"""
        y = np.zeros(len(X))

        for i in range(len(X)):
            y[i] = np.argmin(cityblock(X[i], self.centroids_))

        return y
