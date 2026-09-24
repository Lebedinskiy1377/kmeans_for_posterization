# Image Posterization with K-Medians (from Scratch)

This project reduces the number of colors in an image (posterization) with **K-Medians clustering**, a variant of K-Means that uses **Manhattan (L1) distance** and **per-cluster median centroids**.
It is implemented **from scratch with NumPy** and works directly on pixel data.

## Overview

- Posterizes an image by clustering its pixel colors without supervision
- No ML libraries: only NumPy and `scipy.spatial.distance.cityblock` for the L1 distance
- Updates centroids with the **per-cluster median**, which is more robust to outliers than the mean

## Why this matters

Posterization can:

- reduce the number of colors, e.g. for icons, thumbnails and previews
- produce artistic, stylized images
- serve as a pre-processing step for image classification, compression or edge detection

## Project structure

```text
.
├── main.py          # example script: posterize an image
├── requirements.txt
└── src/
    └── kmeans.py    # ImageKMeans implementation
```

## Installation

```bash
pip install -r requirements.txt
python main.py
```

## Example usage

```python
from src.kmeans import ImageKMeans
from skimage.data import coffee
from skimage.io import imshow

# Load sample image
image = coffee()

# Fit the model on the image pixels
model = ImageKMeans(n_clusters=16, max_iter=100)
model.fit(image)

# Rebuild the image with 16 colors
compressed = model.transform(image)
imshow(compressed)
```

## API

| Method / parameter | Description |
| --- | --- |
| `fit()` | Clusters the RGB pixels of an image |
| `predict()` | Assigns each pixel to the nearest centroid |
| `transform()` | Rebuilds the image from the centroid colors |
| `n_clusters` | Number of colors in the output image |
| `init` | Centroid initialization: `random` or custom |

The L1 distance is computed with `scipy.spatial.distance.cityblock`.
