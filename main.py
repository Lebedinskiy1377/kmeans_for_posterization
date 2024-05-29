from src import ImageKMeans
from skimage.data import coffee
from skimage.io import imshow


def main():
    image = coffee()

    model = ImageKMeans(n_clusters=16, max_iter=100)
    model.fit(image)

    #image_compressed = model.transform(image)
    imshow(image)


main()
