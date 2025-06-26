import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """
    Carica un'immagine 512x512 e la normalizza secondo il preprocessing VGG.
    """
    img = image.load_img(image_path, target_size=(512, 512))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 512, 512, 3)
    img_array = preprocess_input(img_array)        # normalizza (BGR, zero-centered)
    return img_array

def load_feature_extractor():
    """
    Carica il modello VGG16 pre-addestrato (senza fully connected).
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    output_layer = base_model.get_layer('block5_conv3').output
    feature_extractor = Model(inputs=base_model.input, outputs=output_layer)
    return feature_extractor

def extract_features(img_array, feature_extractor):
    """
    Estrae e appiattisce le feature da un'immagine preprocessata.
    """
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten()

def load_and_extract(image_path):
    """
    Pipeline completa: carica immagine, prepara modello, estrae feature.
    """
    img_array = load_and_preprocess_image(image_path)
    feature_extractor = load_feature_extractor()
    features = extract_features(img_array, feature_extractor)
    return img_array, features

def vgg_deprocess(img_tensor):
    """
    De-normalizza immagine VGG (BGR, zero-centered) → RGB [0,255]
    """
    img = img_tensor[0].copy()
    img = img[..., ::-1]  # BGR → RGB
    mean = [103.939, 116.779, 123.68]
    img[..., 0] += mean[0]
    img[..., 1] += mean[1]
    img[..., 2] += mean[2]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def save_rgb_image(img_tensor, filename):
    """
    Salva un'immagine VGG-preprocessata in RGB visivamente corretta.
    """
    img = vgg_deprocess(img_tensor)
    Image.fromarray(img).save(filename)

# ====== ESEMPIO DI USO ======

if __name__ == "__main__":
    path = '/Users/silviurobertplesoiu/Desktop/Unifi/IMAGE VIDEO ANALYSS/Matlab/glaze/Pikachu.png'
    img, feat = load_and_extract(path)
    print("Shape immagine:", img.shape)
    print("Feature vector:", feat.shape)

    # Mostra immagine corretta
    plt.imshow(vgg_deprocess(img))
    plt.title("Immagine RGB corretta")
    plt.axis('off')
    plt.show()

    # Salva immagine
    save_rgb_image(img, "Pikachu_RGB_Correct.png")
    print("✅ Immagine salvata come Pikachu_RGB_Correct.png")
