# analyze_single_image.py

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from numpy.fft import fft2, fftshift
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image

def load_and_preprocess_image(img_path):
    """Carica immagine e preprocessa per VGG."""
    img = keras_image.load_img(img_path, target_size=(512, 512))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Normalizza per VGG
    return img_array

def load_feature_extractor():
    """Carica VGG16 troncato al layer 'block5_conv3'."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
    return model

def extract_features(img, model):
    """Estrai le feature di un'immagine usando il modello."""
    features = model.predict(img, verbose=0)
    return features.flatten()

def compute_metrics(original, cloaked):
    """Calcola PSNR e SSIM."""
    # Denormalizza da [-1,1] a [0,255] per SSIM/PSNR
    orig = ((original[0] + 1) / 2).clip(0, 1)
    cloak = ((cloaked[0] + 1) / 2).clip(0, 1)

    psnr_val = psnr(orig, cloak, data_range=1.0)
    ssim_val = ssim(orig, cloak, data_range=1.0, channel_axis=-1)

    return psnr_val, ssim_val

def show_images(original, cloaked):
    """Visualizza le immagini fianco a fianco."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(((original[0] + 1) / 2).clip(0, 1))
    axs[0].set_title("Originale")
    axs[0].axis('off')

    axs[1].imshow(((cloaked[0] + 1) / 2).clip(0, 1))
    axs[1].set_title("Cloaked")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def show_difference_map(original, cloaked):
    """Visualizza la differenza assoluta normalizzata."""
    diff = np.abs(original[0] - cloaked[0])
    diff /= diff.max()
    plt.imshow(diff)
    plt.title("Mappa delle differenze (assolute)")
    plt.axis('off')
    plt.colorbar()
    plt.show()

def show_fft(image, title):
    """Visualizza lo spettro FFT in scala di grigi."""
    gray = cv2.cvtColor(((image[0] + 1) / 2).astype(np.float32), cv2.COLOR_RGB2GRAY)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f"Spettro FFT - {title}")
    plt.axis('off')
    plt.show()

def analyze_images(original_path, cloaked_path):
    """Analizza e confronta immagine originale e cloaked."""
    # Carica immagini
    original = load_and_preprocess_image(original_path)
    cloaked = load_and_preprocess_image(cloaked_path)

    # Carica modello
    model = load_feature_extractor()

    # Estrai feature
    features_orig = extract_features(original, model)
    features_cloak = extract_features(cloaked, model)

    # Calcola metriche
    psnr_val, ssim_val = compute_metrics(original, cloaked)
    l2_distance = np.linalg.norm(features_orig - features_cloak)

    # Output
    print(f"📏 PSNR: {psnr_val:.2f} dB")
    print(f"🔍 SSIM: {ssim_val:.4f}")
    print(f"🧠 L2 distance tra feature: {l2_distance:.4f}")

    # Visualizza
    show_images(original, cloaked)
    show_difference_map(original, cloaked)
    show_fft(original, "Originale")
    show_fft(cloaked, "Cloaked")

# Esempio di utilizzo diretto
if __name__ == "__main__":
    # Metti i path delle immagini da confrontare
    original_path = "Pikachu.png"
    cloaked_path = "Pikachu-protected-intensity-DEFAULT-V2.png"

    analyze_images(original_path, cloaked_path)
