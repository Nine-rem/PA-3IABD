#!/usr/bin/env python3
import os
import ctypes
import platform
import numpy as np

# Configuration de base
CRATE_NAME = "ml_library"
if platform.system() == "Windows":
    lib_fname = f"{CRATE_NAME}.dll"
else:
    lib_fname = f"lib{CRATE_NAME}.so"

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_library"))
lib_path = os.path.join(root, "target", "release", lib_fname)

print(f">>> Library path: {lib_path}, exists? {os.path.exists(lib_path)}")
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Bibliothèque introuvable : {lib_path}")

# Charger la bibliothèque
lib = ctypes.CDLL(lib_path)

# Configuration de la fonction train_pmc
lib.train_pmc.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x_data
    ctypes.c_size_t,                 # n_samples  
    ctypes.c_size_t,                 # n_features
    ctypes.POINTER(ctypes.c_float),  # y_data
    ctypes.POINTER(ctypes.c_size_t), # hidden_sizes
    ctypes.c_size_t,                 # n_hidden_layers
    ctypes.c_int,                    # hidden_act
    ctypes.c_int,                    # output_act
    ctypes.c_size_t,                 # n_iters
    ctypes.POINTER(ctypes.c_float),  # weights_out
    ctypes.c_float,                  # learning_rate
    ctypes.c_int                     # is_classification
]
lib.train_pmc.restype = None

def train_pmc(X, Y, hidden_sizes=[10, 10], hidden_act=0, output_act=0, n_iters=500, learning_rate=0.1, is_classification=True):
    n_samples, n_features = X.shape
    
    print(f">>> [PY] train_pmc called with X.shape={X.shape}, hidden_sizes={hidden_sizes}, hidden_act={hidden_act}, output_act={output_act}, n_iters={n_iters}")
    
    # Configuration des layers
    layers = [n_features] + hidden_sizes + [1]
    print(f"Architecture: {layers}")
    
    # Calcul de l'espace nécessaire pour les poids
    weights_len = sum((layers[i] + 1) * layers[i + 1] for i in range(len(layers) - 1))
    
    # Préparer les données
    x_flat = X.flatten().astype(np.float32)
    y_flat = Y.astype(np.float32)
    hidden_array = np.array(hidden_sizes, dtype=np.uintp)
    weights_out = np.zeros(weights_len, dtype=np.float32)
    
    # Appeler la fonction Rust
    lib.train_pmc(
        x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_samples,
        n_features,
        y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        hidden_array.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        len(hidden_sizes),
        hidden_act,
        output_act,
        n_iters,
        weights_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        learning_rate,
        1 if is_classification else 0
    )
    
    return weights_out, layers

# Test rapide : Classification XOR
print("=== Test XOR ===")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y_xor = np.array([-1, 1, 1, -1], dtype=np.float32)

try:
    weights, layers = train_pmc(X_xor, Y_xor, hidden_sizes=[4, 4], n_iters=1000)
    print("V/ Entraînement XOR réussi!")
    print(f"Poids entraînés: {len(weights)} paramètres")
except Exception as e:
    print(f"X Erreur XOR: {e}")

# Test rapide : Classification linéaire
print("\n=== Test Classification Linéaire ===")
X_lin = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
Y_lin = np.array([-1, 1, 1], dtype=np.float32)

try:
    weights, layers = train_pmc(X_lin, Y_lin, hidden_sizes=[5, 5], n_iters=500)
    print("V/ Entraînement linéaire réussi!")
    print(f"Poids entraînés: {len(weights)} paramètres")
except Exception as e:
    print(f"X Erreur linéaire: {e}")

print("\n=== Test Régression ===")
X_reg = np.array([[1], [2]], dtype=np.float32)
Y_reg = np.array([2, 3], dtype=np.float32)

try:
    weights, layers = train_pmc(X_reg, Y_reg, hidden_sizes=[10, 10], n_iters=500, is_classification=False, output_act=2)
    print("V/ Entraînement régression réussi!")
    print(f"Poids entraînés: {len(weights)} paramètres")
except Exception as e:
    print(f"X Erreur régression: {e}")

print("\nV/ Tests rapides PMC terminés!")
