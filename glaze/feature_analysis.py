# feature_analysis.py

import numpy as np
from load_and_extract import extract_features

def compute_feature_distance(original_features, cloaked_img, feature_extractor, verbose=True):
    """
    Calcola la distanza L2 tra le feature originali e quelle dell'immagine cloaked.

    Args:
        original_features (np.ndarray): vettore feature originale
        cloaked_img (np.ndarray): immagine perturbata (1, 512, 512, 3)
        feature_extractor (Model): modello VGG tagliato
        verbose (bool): se True stampa il risultato

    Returns:
        float: distanza euclidea tra le due rappresentazioni
    """
    cloaked_features = extract_features(cloaked_img, feature_extractor)
    distance = np.linalg.norm(cloaked_features - original_features)

    if verbose:
        print(f"📏 L2 distance tra original e cloaked features: {distance:.4f}")
    return distance
