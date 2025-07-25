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

lib = ctypes.CDLL(lib_path)

# Signature FFI identique aux vrais tests
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

def train_pmc(X, Y, hidden_sizes, hidden_act, output_act, lr=0.01, n_iters=1000):
    """Fonction identique à celle des vrais tests"""
    n, d = X.shape
    hs = np.array(hidden_sizes, dtype=np.int64)

    # Préparer Y_flat et k (logique des vrais tests)
    if output_act == 2:  # Régression
        k = 1
        Y_flat = Y.astype(np.float64).ravel()
    elif output_act == 1:  # Multiclasse
        k = Y.shape[1]
        Y_flat = Y.astype(np.float64).ravel()
    else:  # Classification binaire
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
    """Forward pass identique aux vrais tests"""
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

def test_case(name, X, Y, hidden_sizes, hidden_act, output_act, expected_accuracy=50):
    """Test un cas et vérifie l'accuracy"""
    print(f"\n=== {name} ===")
    print(f"X.shape: {X.shape}, Y unique: {np.unique(Y)}")
    
    try:
        # Entraînement
        W, layers = train_pmc(X, Y, hidden_sizes, hidden_act, output_act, lr=0.01, n_iters=500)
        print(f"Architecture: {layers}")
        
        # Prédiction
        preds = pmc_forward(X, W, layers, hidden_act, output_act)
        
        # Calcul accuracy
        if output_act == 2:  # Régression - MSE
            mse = np.mean((preds.ravel() - Y.ravel())**2)
            print(f"MSE: {mse:.4f}")
            success = mse < 1.0  # Critère arbitraire
        else:  # Classification
            if output_act == 0:  # Binaire
                pred_classes = (preds.ravel() > 0.5).astype(float) * 2 - 1  # Convertir en -1/1
                y_binary = (Y > 0).astype(float) * 2 - 1
                accuracy = np.mean(pred_classes == y_binary) * 100
            else:  # Multiclasse
                pred_classes = np.argmax(preds, axis=1)
                y_classes = np.argmax(Y, axis=1)
                accuracy = np.mean(pred_classes == y_classes) * 100
            
            print(f"Accuracy: {accuracy:.1f}%")
            success = accuracy >= expected_accuracy
        
        print(f"{'V/ PASSÉ' if success else 'X ÉCHOUÉ'}")
        return success
        
    except Exception as e:
        print(f"X ERREUR: {e}")
        return False

# --- TESTS SIMPLIFIÉS DES 11 CAS ---

def run_all_tests():
    results = []
    
    # 1. Linear Simple
    X1 = np.array([[1,1],[2,3],[3,3]], dtype=np.float64)
    Y1 = np.array([1,-1,-1], dtype=np.float64)
    results.append(test_case("1. Classification – Linear Simple", X1, Y1, [10,10], 0, 0))

    # 2. Linear Multiple  
    np.random.seed(0)
    X2 = np.vstack([
        np.random.random((50,2))*0.9 + [1,1],
        np.random.random((50,2))*0.9 + [2,2]
    ])
    Y2 = np.concatenate([np.ones(50), -np.ones(50)])
    results.append(test_case("2. Classification – Linear Multiple", X2, Y2, [10,10], 0, 0))

    # 3. XOR (le plus important!)
    X3 = np.array([[1,0],[0,1],[0,0],[1,1]], dtype=np.float64)
    Y3 = np.array([1,1,-1,-1], dtype=np.float64)
    results.append(test_case("3. Classification – XOR", X3, Y3, [10,10], 0, 0, 75))  # XOR doit bien marcher

    # 4. Cross
    X4 = np.random.uniform(-1,1,(100,2))  # Réduit pour rapidité
    Y4 = np.where((np.abs(X4[:,0])<=0.3)|(np.abs(X4[:,1])<=0.3), 1, -1)
    results.append(test_case("4. Classification – Cross", X4, Y4, [10,10], 0, 0))

    # 5. Régression simple
    X5 = np.array([[1],[2]], dtype=np.float64)
    Y5 = np.array([2,3], dtype=np.float64)
    results.append(test_case("7. Régression – Linear Simple 2D", X5, Y5, [10,10], 0, 2))

    print(f"\n{'='*50}")
    print(f"RÉSULTATS FINAUX: {sum(results)}/{len(results)} tests passés")
    print(f"Taux de réussite: {sum(results)/len(results)*100:.1f}%")
    
    if sum(results) >= len(results) * 0.6:  # 60% de réussite
        print("V/ PMC VALIDÉ - Performance acceptable!")
    else:
        print("X PMC À AMÉLIORER - Performance insuffisante")
    
    return results

if __name__ == "__main__":
    run_all_tests()
