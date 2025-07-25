import os
import ctypes
import platform
import numpy as np

# Configuration
CRATE_NAME = "ml_library"
if platform.system() == "Windows":
    lib_fname = f"{CRATE_NAME}.dll"
else:
    lib_fname = f"lib{CRATE_NAME}.so"

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_library"))
lib_path = os.path.join(root, "target", "debug", lib_fname)

lib = ctypes.CDLL(lib_path)

# Signature FFI pour train_pmc
lib.train_pmc.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # X
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # Y
    ctypes.c_size_t,  # n_samples
    ctypes.c_size_t,  # n_features
    ctypes.c_size_t,  # n_hidden_layers
    np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),    # hidden_sizes
    ctypes.c_size_t,  # n_outputs
    ctypes.c_size_t,  # hidden_act
    ctypes.c_size_t,  # output_act
    ctypes.c_double,  # lr
    ctypes.c_size_t,  # n_iters
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # out weights
]
lib.train_pmc.restype = None

def test_xor():
    print("=== TEST XOR PMC ===")
    
    # Données XOR
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    Y = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)  # XOR logique
    
    print(f"Données d'entrée X:\n{X}")
    print(f"Labels Y: {Y}")
    
    # Configuration du réseau
    hidden_sizes = np.array([4], dtype=np.int64)  # 1 couche cachée de 4 neurones
    layers = [2, 4, 1]  # 2 entrées, 4 cachés, 1 sortie
    
    # Calcul taille des poids
    weights_size = sum((layers[i] + 1) * layers[i+1] for i in range(len(layers)-1))
    weights_out = np.zeros(weights_size, dtype=np.float64)
    
    print(f"Architecture: {layers}")
    print(f"Taille poids: {weights_size}")
    
    # Entraînement
    print("Entraînement en cours...")
    lib.train_pmc(
        X, Y,                    # Données
        4, 2,                    # 4 échantillons, 2 features
        1, hidden_sizes,         # 1 couche cachée
        1,                       # 1 sortie
        0,                       # hidden_act (ignoré)
        0,                       # output_act: sigmoid pour classification binaire
        0.01,                    # learning rate (plus petit)
        5000,                    # epochs (plus d'itérations)
        weights_out
    )
    
    print(f"Poids obtenus: {weights_out}")
    
    # Test de prédiction manuel (simulation du forward pass)
    def predict_xor(x1, x2, weights):
        # Reconstruction des poids
        # Couche1: 2 entrées + 1 biais -> 4 sorties = (2+1)*4 = 12 poids
        # Couche2: 4 entrées + 1 biais -> 1 sortie  = (4+1)*1 = 5 poids
        
        w1 = weights[:12].reshape(4, 3)  # 4 neurones x (2 entrées + 1 biais)
        w2 = weights[12:].reshape(1, 5)   # 1 neurone x (4 entrées + 1 biais)
        
        # Forward pass
        x_in = np.array([x1, x2, 1.0])  # Ajouter biais
        
        # Couche cachée
        z1 = w1.dot(x_in)
        a1 = 1.0 / (1.0 + np.exp(-z1))  # sigmoid
        
        # Couche sortie  
        x_hidden = np.concatenate([a1, [1.0]])  # Ajouter biais
        z2 = w2.dot(x_hidden)
        a2 = 1.0 / (1.0 + np.exp(-z2))  # sigmoid
        
        return a2[0]
    
    # Test sur tous les cas XOR
    print("\n=== RÉSULTATS XOR ===")
    correct = 0
    for i, (x1, x2) in enumerate([[0,0], [0,1], [1,0], [1,1]]):
        pred = predict_xor(x1, x2, weights_out)
        expected = Y[i]
        pred_class = 1.0 if pred >= 0.5 else -1.0
        is_correct = pred_class == expected
        correct += is_correct
        
        print(f"XOR({x1},{x2}): pred={pred:.3f} -> {pred_class}, attendu={expected}, {'V/' if is_correct else 'X'}")
    
    accuracy = correct / 4.0 * 100
    print(f"\n ACCURACY: {accuracy:.1f}% ({correct}/4)")
    
    if accuracy >= 50:
        print("V/ PMC fonctionne correctement!")
    else:
        print("X PMC a besoin d'améliorations...")
    
    return accuracy

if __name__ == "__main__":
    test_xor()
