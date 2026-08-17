import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.utils import QuantumInstance
from qiskit import Aer
from joblib import dump

# Fonction d'extraction des caractéristiques de l'image
def dummy_embedding(image):
    image = cv2.resize(image, (32, 32))  # Redimensionner l'image
    return image.flatten()[:128] / 255.0  # Normaliser les pixels

# Fonction pour charger les images et leurs labels
def load_images(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, label)
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            emb = dummy_embedding(img)
            X.append(emb)
            y.append(label)
    return np.array(X), np.array(y)

# Fonction pour entraîner le classificateur quantique
def train_quantum_classifier(X, y):
    le = LabelEncoder()  # Encoder les labels en entiers
    y_encoded = le.fit_transform(y)
    
    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Application de PCA pour réduire les dimensions
    pca = PCA(n_components=8)  # Augmenter le nombre de composants pour améliorer la qualité des features
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)  # Appliquer la même transformation au jeu de test

    # Définir la carte de caractéristiques quantiques et l'ansatz
    feature_map = ZZFeatureMap(feature_dimension=8, reps=2)  # Ajuster feature_dimension à 8 pour correspondre aux dimensions de PCA
    ansatz = TwoLocal(8, ['ry', 'rz'], 'cz', reps=2)  # Deux rotations pour correspondre à la dimension des features
    quantum_instance = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=1024)

    # Initialiser le classificateur quantique VQC
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=None, quantum_instance=quantum_instance)
    
    # Entraîner le classificateur
    vqc.fit(X_train_pca, y_train)

    # Sauvegarder le modèle PCA et le classificateur quantique pour utilisation future
    dump(pca, "../quantum_server/pca.joblib")
    dump(vqc, "../quantum_server/quantum_model.joblib")

# Point d'entrée du script
if __name__ == '__main__':
    data_dir = "../dataset"  # Répertoire contenant les images
    X, y = load_images(data_dir)  # Charger les images et les labels
    train_quantum_classifier(X, y)  # Entraîner le classificateur quantique
