import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def proses_citra(path='ayam.jpg'):
    # 1. Baca gambar pakai imageio
    img = iio.imread(path)

    # Kalau gambarnya berwarna (RGB), ubah ke grayscale pakai numpy
    if img.ndim == 3:
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img

    # 2. Kurangi noise pakai Gaussian filter dari scipy.ndimage
    blur = ndimage.gaussian_filter(gray, sigma=1.5)

    # 3. Penajaman citra (sharpening / unsharp masking)
    enhanced = np.clip(
        gray.astype(float) + 1.5 * (gray.astype(float) - blur), 0, 255
    ).astype(np.uint8)

    # 4. Tampilkan perbandingan gambar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Citra Grayscale Asli')
    axes[0].axis('off')

    axes[1].imshow(blur, cmap='gray')
    axes[1].set_title('Hasil Gaussian Blur')
    axes[1].axis('off')

    axes[2].imshow(enhanced, cmap='gray')
    axes[2].set_title('Hasil Penajaman (Enhanced)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# Panggil fungsinya
proses_citra('ayam.jpg')