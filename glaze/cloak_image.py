# cloak_image.py

import numpy as np
import tensorflow as tf

def generate_perturbation(img, features_original, model, epsilon=5/255, steps=30, alpha=1/255):
    """
    Genera una piccola perturbazione per l'immagine che altera le feature.
    
    Args:
        img: immagine preprocessata, shape (1, 512, 512, 3)
        features_original: vettore di feature originali (np.ndarray 1D)
        model: modello feature extractor (es. VGG tagliato)
        epsilon: massimo valore di variazione pixel-wise (default 5/255)
        steps: numero iterazioni ottimizzazione
        alpha: passo di aggiornamento

    Returns:
        perturbazione finale (np.ndarray) da sommare all'immagine
    """
    # Converti features originali in tensor
    target = tf.convert_to_tensor(features_original, dtype=tf.float32)

    # Inizializza perturbazione come variabile ottimizzabile
    perturbation = tf.Variable(tf.zeros_like(img), trainable=True)

    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha)

    for step in range(steps):
        with tf.GradientTape() as tape:
            adv_img = img + perturbation
            adv_features = model(adv_img, training=False)
            adv_features = tf.reshape(adv_features, [-1])
            loss = tf.reduce_mean(tf.square(adv_features - target))  # L2 distance

        grads = tape.gradient(loss, [perturbation])
        optimizer.apply_gradients(zip(grads, [perturbation]))

        # Clip perturbazione per non superare epsilon
        perturbation.assign(tf.clip_by_value(perturbation, -epsilon, epsilon))

    return perturbation.numpy()

def apply_perturbation(img, perturbation):
    """
    Applica la perturbazione all'immagine e clippa nel range valido VGG.

    Args:
        img: immagine originale preprocessata
        perturbation: array da sommare (stessa shape)

    Returns:
        immagine cloaked (np.ndarray)
    """
    adv_img = img + perturbation
    adv_img = tf.clip_by_value(adv_img, -1.0, 1.0)  # range compatibile con VGG
    return adv_img.numpy()

def cloak_image(img, features_original, model, epsilon=5/255, steps=30, alpha=1/255):
    """
    Funzione completa: genera il cloak e lo applica.

    Args:
        img: immagine preprocessata (1, 512, 512, 3)
        features_original: vettore feature originali
        model: feature extractor

    Returns:
        immagine cloaked
    """
    perturbation = generate_perturbation(img, features_original, model, epsilon, steps, alpha)
    img_cloaked = apply_perturbation(img, perturbation)
    return img_cloaked

# Esempio di test (da usare solo per debug)
if __name__ == "__main__":
    from load_and_extract import load_and_extract, load_feature_extractor

    path = "immagine.jpg"  # sostituire con immagine 512x512
    img, feat = load_and_extract(path)
    model = load_feature_extractor()

    img_cloaked = cloak_image(img, feat, model)
    print("Cloaked image generata con successo.")
