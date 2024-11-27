import imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Baca gambar
image = imageio.imread("C:\gambar\grayscale.jpg")  # Gambar grayscale

# Ekualisasi histogram
def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()  # Distribusi kumulatif
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')
    img_equalized = cdf_normalized[img.astype('uint8')]
    return img_equalized

equalized_image = histogram_equalization(image)

# Terapkan filter Gaussian untuk menghaluskan gambar
smoothed_image = ndimage.gaussian_filter(equalized_image, sigma=1)

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Gaussian Filtered')
plt.imshow(smoothed_image, cmap='gray')

plt.tight_layout()
plt.show()
