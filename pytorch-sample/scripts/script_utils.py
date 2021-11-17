from imageio import imread, imwrite

# Create a download function
def download_image(url, path):
    data = imread(url)
    imwrite(path / url.split("/")[-1], data)


