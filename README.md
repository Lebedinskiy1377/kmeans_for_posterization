K-Means for Posterization
A Python library for image posterization using the K-Means clustering algorithm to reduce the number of colors in an image, creating a stylized, compressed version.
Features

K-Means Clustering: Implements K-Means to cluster image pixels based on RGB values.
Image Posterization: Reduces the color palette of an image to a specified number of clusters.
Customizable Parameters: Supports adjustable number of clusters, maximum iterations, and random state for reproducibility.
Efficient Processing: Uses numpy and scipy for fast computation and skimage for image handling.

Tech Stack

Python 3.9+
numpy
scipy
scikit-image

Installation

Clone the repository:git clone https://github.com/Lebedinskiy1377/kmeans_for_posterization.git
cd kmeans_for_posterization


Install dependencies:pip install -r requirements.txt



Usage
1. Posterizing an Image
from src.kmeans import ImageKMeans
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

# Load an image
image = imread("data/sample_image.jpg")

# Initialize and fit the K-Means model
model = ImageKMeans(n_clusters=16, max_iter=100, random_state=42)
model.fit(image)

# Transform the image to its posterized version
image_compressed = model.transform(image)

# Display the result
imshow(image_compressed)
plt.show()

Example Input Data
The input should be an RGB image (numpy array of shape (height, width, 3)) with pixel values in the range [0, 255]. You can use images in formats like .jpg or .png.
Example:

Place an image (e.g., sample_image.jpg) in the data/ directory or use a sample from skimage.data (e.g., coffee()).

Example Output
Running the above code with n_clusters=16 will produce a posterized image with only 16 unique colors, creating a stylized effect.
Project Structure
├── data/               # Directory for sample images (optional)
├── src/                # Source code
│   ├── kmeans.py       # K-Means clustering implementation for image posterization
├── main.py             # Example script to demonstrate usage
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

Requirements
Install dependencies using:
pip install numpy scipy scikit-image matplotlib

Notes

The main.py script currently displays the original image without applying the posterization transformation. To see the posterized result, uncomment the line image_compressed = model.transform(image) and use imshow(image_compressed).
The distance metric used in kmeans.py is Manhattan (cityblock) distance. You can modify _assign_centroids to use Euclidean distance if preferred.
