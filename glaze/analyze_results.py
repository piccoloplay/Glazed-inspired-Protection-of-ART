import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numpy.fft import fft2, fftshift
import cv2
def show_images_side_by_side(original, cloaked, title1="Originale", title2="Cloaked"):
    """Mostra le due immagini fianco a fianco."""
    img1 = original[0] if original.ndim == 4 else original
    img2 = cloaked[0] if cloaked.ndim == 4 else cloaked

    img1 = ((img1 + 1) / 2).clip(0, 1)
    img2 = ((img2 + 1) / 2).clip(0, 1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img1)
    axs[0].set_title(title1)
    axs[0].axis('off')

    axs[1].imshow(img2)
    axs[1].set_title(title2)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()



def show_difference_map(original, cloaked):
    """Mostra la mappa delle differenze assolute tra le immagini."""
    diff = np.abs(original[0] - cloaked[0])  # shape: (512, 512, 3)
    diff_norm = diff / (diff.max() + 1e-8)   # normalizza
    plt.imshow(diff_norm)
    plt.title("Mappa delle differenze (assolute)")
    plt.axis('off')
    plt.colorbar()
    plt.show()

def show_fft(image, title="FFT"):
    """Mostra lo spettro delle frequenze (FFT) di un'immagine in scala di grigi."""
    gray = cv2.cvtColor(((image[0] + 1) / 2).astype(np.float32), cv2.COLOR_RGB2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f"Spettro FFT - {title}")
    plt.axis('off')
    plt.show()

def compute_metrics(original, cloaked):
    """Calcola PSNR e SSIM tra immagini originali e cloaked."""
    orig = ((original[0] + 1) / 2).astype(np.float32)
    cloak = ((cloaked[0] + 1) / 2).astype(np.float32)

    psnr_val = psnr(orig, cloak, data_range=1.0)
    ssim_val = ssim(orig, cloak, data_range=1.0, channel_axis=-1)  # skimage>=0.19

    print(f"📏 PSNR: {psnr_val:.2f} dB")
    print(f"🔍 SSIM: {ssim_val:.4f}")
