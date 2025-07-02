# K-Means Posterization of Images (from Scratch)

This project implements **K-Means clustering** for image posterization — a technique that reduces the number of colors in an image.  
The model is implemented **from scratch using NumPy**, and works directly on pixel-level data using **Manhattan (L1) distance** and **medoid-style centroids (median)**.

---

## Overview

- Posterizes an image by reducing color diversity via unsupervised clustering (K-means)
- Implemented without any external ML libraries
- Uses `scipy.spatial.distance.cityblock` (L1 distance) instead of Euclidean
- Centroids are updated using **per-cluster median** (more robust than mean)

---

## Why this matters

Posterization can:
- Reduce image size (e.g., for icons, thumbnails, previews)
- Produce artistic, stylized images
- Be used in pre-processing pipelines for image classification, compression, or edge detection

---

## Project Structure

├── main.py # Example script: compress image using K-means
├── src/
│ └── kmeans.py # Core implementation of ImageKMeans

## Example usage

```python
from src.kmeans import ImageKMeans
from skimage.data import coffee
from skimage.io import imshow

# Load sample image
image = coffee()

# Train custom K-Means
model = ImageKMeans(n_clusters=16, max_iter=100)
model.fit(image)

# Generate compressed image
compressed = model.transform(image)
imshow(compressed)
```

## Features
Feature	Description
fit()	Trains K-means on RGB image pixels
predict()	Assigns each pixel to nearest cluster centroid
transform()	Reconstructs compressed image using centroids
n_clusters	Number of unique colors in final image
init	Centroid init strategy: random or custom
cityblock()	Uses Manhattan distance (L1) for clustering
