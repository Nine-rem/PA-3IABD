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

def train_pmc(X, Y, hidden_sizes, hidden_act, output_act, lr=0.01, n_iters=2000):
    n, d = X.shape
    hs = np.array(hidden_sizes, dtype=np.int64)

    # préparer Y_flat et k
    if output_act == 2:
        k = 1
        Y_flat = Y.astype(np.float64).ravel()
    elif output_act == 1:
        k = Y.shape[1]
        Y_flat = Y.astype(np.float64).ravel()
    else:
        k = 1
        Yb = (Y > 0).astype(np.float64)
        Y_flat = Yb.ravel()

    layers = [d] + hidden_sizes + [k]
    total = sum((layers[i] + 1) * layers[i+1] for i in range(len(layers)-1))
    W = np.zeros(total, dtype=np.float64)

    print(f">>> [PY] train_pmc called with X.shape={X.shape}, hidden_sizes={hidden_sizes}")

    lib.train_pmc(
        X.astype(np.float64),
        Y_flat,
        n, d, len(hidden_sizes), hs,
        k, hidden_act, output_act,
        lr, n_iters,
        W
    )
    return W, layers

def pmc_forward(X, W, layers, hidden_act, output_act):
    weights = []
    idx = 0
    for i in range(len(layers)-1):
        out_sz, in_sz = layers[i+1], layers[i] + 1
        cnt = out_sz * in_sz
        mat = W[idx:idx+cnt].reshape(out_sz, in_sz)
        weights.append(mat)
        idx += cnt

    H = X.copy()
    for i, Wm in enumerate(weights):
        H = np.hstack([np.ones((H.shape[0],1)), H])
        Z = H.dot(Wm.T)
        if i < len(weights)-1:
            H = np.maximum(0, Z) if hidden_act == 0 else np.tanh(Z)
        else:
            if output_act == 0:
                H = 1/(1+np.exp(-Z))
            elif output_act == 1:
                e = np.exp(Z - Z.max(axis=1, keepdims=True))
                H = e / e.sum(axis=1, keepdims=True)
            else:
                H = Z
    return H

def test_xor():
    """Test XOR - un problème classique non-linéaire"""
    print("\n=== Test XOR AMÉLIORÉ ===")
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    Y = np.array([-1,1,1,-1], dtype=np.float64)  # XOR avec labels -1,1
    
    print("Données d'entrée:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {Y[i]}")
    
    # Entraînement avec learning rate original
    W, layers = train_pmc(X, Y, [10,10], 0, 0, lr=0.01, n_iters=5000)
    
    # Test des prédictions
    preds = pmc_forward(X, W, layers, 0, 0)
    print("\nPrédictions finales:")
    for i in range(len(X)):
        pred_prob = preds[i,0]
        pred_class = 1 if pred_prob > 0.5 else -1
        correct = "✓" if pred_class == Y[i] else "✗"
        print(f"  {X[i]} -> prob: {pred_prob:.4f}, pred: {pred_class:2.0f}, true: {Y[i]:2.0f} {correct}")
    
    # Calcul de l'accuracy
    pred_classes = np.where(preds > 0.5, 1, -1).ravel()
    accuracy = np.mean(pred_classes == Y)
    print(f"\nAccuracy: {accuracy:.1%}")
    
    if accuracy >= 1.0:
        print("V/ XOR résolu parfaitement!")
    elif accuracy >= 0.75:
        print("V/ XOR résolu avec succès!")
    else:
        print("X XOR non résolu.")
        
    return accuracy >= 0.75

if __name__ == "__main__":
    test_xor()
