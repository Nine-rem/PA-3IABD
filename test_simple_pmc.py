import os
import ctypes
import platform
import numpy as np

# Configuration de la bibliothèque
CRATE_NAME = "ml_library"

if platform.system() == "Windows":
    lib_fname = f"{CRATE_NAME}.dll"
elif platform.system() == "Darwin":
    lib_fname = f"lib{CRATE_NAME}.dylib"
else:
    lib_fname = f"lib{CRATE_NAME}.so"

# Chemin vers la bibliothèque compilée
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_library"))
lib_path = os.path.join(root, "target", "debug", lib_fname)
print(">>> Library path:", lib_path, "exists?", os.path.exists(lib_path))

if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Bibliothèque introuvable : {lib_path}")

lib = ctypes.CDLL(lib_path)

# Test simple de la fonction train_pmc
print("🧪 Test simple de train_pmc")

# Signature FFI
lib.train_pmc.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # X
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # Y
    ctypes.c_size_t,  # n_samples
    ctypes.c_size_t,  # n_features
    ctypes.c_size_t,  # n_hidden_layers
    np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),    # hidden_sizes
    ctypes.c_size_t,  # n_outputs
    ctypes.c_size_t,  # hidden_act (0=ReLU,1=Tanh)
    ctypes.c_size_t,  # output_act (0=Sigmoid,1=Softmax,2=Linear)
    ctypes.c_double,  # lr
    ctypes.c_size_t,  # n_iters
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # out weights
]
lib.train_pmc.restype = None

try:
    # Test avec des données très simples
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    Y = np.array([0.0, 1.0], dtype=np.float64)
    hidden_sizes = np.array([3], dtype=np.int64)
    
    # Calculer la taille des poids
    layers = [2, 3, 1]
    weights_size = sum((layers[i] + 1) * layers[i+1] for i in range(len(layers)-1))
    weights_out = np.zeros(weights_size, dtype=np.float64)
    
    print(f"Données: X.shape={X.shape}, Y.shape={Y.shape}")
    print(f"Couches: {layers}, taille poids: {weights_size}")
    
    # Appel de la fonction
    lib.train_pmc(
        X, Y,
        2, 2, 1, hidden_sizes,
        1, 0, 0, 0.01, 100,
        weights_out
    )
    
    print("✅ Fonction train_pmc appelée avec succès!")
    print(f"Poids retournés: {weights_out[:10]}...")  # Afficher les 10 premiers
    
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
