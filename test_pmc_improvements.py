import os
import ctypes
import platform
import numpy as np
import matplotlib.pyplot as plt

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
lib_path = os.path.join(root, "target", "release", lib_fname)
print(">>> Library path:", lib_path, "exists?", os.path.exists(lib_path))

if not os.path.exists(lib_path):
    # Essayer avec debug si release n'existe pas
    lib_path = os.path.join(root, "target", "debug", lib_fname)
    print(">>> Trying debug path:", lib_path, "exists?", os.path.exists(lib_path))

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

def train_pmc(X, Y, hidden_sizes, hidden_act, output_act, lr=0.01, n_iters=3000):
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

    print(f">>> [PY] train_pmc called with X.shape={X.shape}, hidden_sizes={hidden_sizes}, hidden_act={hidden_act}, output_act={output_act}, n_iters={n_iters}")

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
    print("\n=== Test XOR ===")
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    Y = np.array([0,1,1,0], dtype=np.float64)  # XOR
    
    print("Données d'entrée:")
    print("X =", X)
    print("Y =", Y)
    
    # Entraînement avec paramètres équilibrés pour XOR
    W, layers = train_pmc(X, Y, [10,10], 0, 0, lr=0.1, n_iters=5000)
    
    # Test des prédictions avec activation ReLU
    preds = pmc_forward(X, W, layers, 0, 0)
    print("\nPrédictions:")
    for i in range(len(X)):
        print(f"X{i}: {X[i]} -> pred: {preds[i,0]:.4f}, true: {Y[i]}")
    
    # Calcul de l'accuracy
    pred_classes = (preds > 0.5).astype(int).ravel()
    accuracy = np.mean(pred_classes == Y.astype(int))
    print(f"Accuracy: {accuracy:.2%}")
    
    return accuracy > 0.8  # Au moins 80% de réussite

def test_simple_classification():
    """Test classification binaire simple"""
    print("\n=== Test Classification Binaire Simple ===")
    np.random.seed(42)
    
    # Créer deux groupes séparables
    n_samples = 100
    X1 = np.random.normal([2, 2], 0.5, (n_samples//2, 2))
    X2 = np.random.normal([-2, -2], 0.5, (n_samples//2, 2))
    X = np.vstack([X1, X2])
    Y = np.hstack([np.ones(n_samples//2), -np.ones(n_samples//2)])
    
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Entraînement avec paramètres plus conservateurs
    W, layers = train_pmc(X, Y, [12], 0, 0, lr=0.05, n_iters=3000)
    
    # Test des prédictions
    preds = pmc_forward(X, W, layers, 0, 0)
    pred_classes = np.where(preds > 0.5, 1, -1).ravel()
    accuracy = np.mean(pred_classes == Y)
    print(f"Accuracy: {accuracy:.2%}")
    
    return accuracy > 0.85

def test_multiclass():
    """Test classification multiclasse"""
    print("\n=== Test Classification Multiclasse ===")
    np.random.seed(42)
    
    # Créer 3 classes
    n_per_class = 30
    X1 = np.random.normal([2, 2], 0.3, (n_per_class, 2))
    X2 = np.random.normal([-2, 2], 0.3, (n_per_class, 2))
    X3 = np.random.normal([0, -2], 0.3, (n_per_class, 2))
    
    X = np.vstack([X1, X2, X3])
    Y = np.array([[1,0,0]]*n_per_class + [[0,1,0]]*n_per_class + [[0,0,1]]*n_per_class, dtype=np.float64)
    
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Entraînement avec learning rate adapté
    W, layers = train_pmc(X, Y, [15,10], 0, 1, lr=0.1, n_iters=4000)
    
    # Test des prédictions
    preds = pmc_forward(X, W, layers, 0, 1)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(Y, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    print(f"Accuracy: {accuracy:.2%}")
    
    return accuracy > 0.80

if __name__ == "__main__":
    print("🧪 Tests des améliorations du PMC")
    print("=" * 50)
    
    tests = [
        ("XOR", test_xor),
        ("Classification binaire", test_simple_classification),
        ("Classification multiclasse", test_multiclass),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'RÉUSSI' if success else 'ÉCHOUÉ'}")
        except Exception as e:
            print(f"❌ {test_name}: ERREUR - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Résultats finaux:")
    successes = sum(1 for _, success in results if success)
    print(f"{successes}/{len(results)} tests réussis")
    
    if successes == len(results):
        print("🎉 Toutes les améliorations fonctionnent correctement!")
    else:
        print("⚠️  Certains tests ont échoué, vérifiez l'implémentation.")
