#!/usr/bin/env python3
"""
Test complet PMC - 11 cas sans visualisation (pour vérifier les erreurs)
Compatible avec cas_de_test_pmc.py mais sans matplotlib
"""
import os
import ctypes
import platform
import numpy as np
import sys

print("="*60)
print("TEST PMC - BATTERIE COMPLÈTE DES 11 CAS")
print("="*60)

# Configuration de base
CRATE_NAME = "ml_library" 
if platform.system() == "Windows":
    lib_fname = f"{CRATE_NAME}.dll"
else:
    lib_fname = f"lib{CRATE_NAME}.so"

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_library")) 
lib_path = os.path.join(root, "target", "release", lib_fname)

print(f"Library path: {lib_path}")
print(f"Exists: {os.path.exists(lib_path)}")

if not os.path.exists(lib_path):
    print("ERREUR: DLL introuvable!")
    sys.exit(1)

try:
    lib = ctypes.CDLL(lib_path)
    print("Bibliothèque chargée avec succès")
except Exception as e:
    print(f"ERREUR chargement: {e}")
    sys.exit(1)

# Signature FFI (exactement comme cas_de_test_pmc.py)
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

def train_pmc(X, Y, hidden_sizes, hidden_act, output_act, lr=0.01, n_iters=5000):
    n, d = X.shape
    hs = np.array(hidden_sizes, dtype=np.int64)

    # Préparer Y_flat et k
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
        X.astype(np.float64), Y_flat, n, d, len(hidden_sizes), hs,
        k, hidden_act, output_act, lr, n_iters, W
    )
    return W, layers

def pmc_forward(X, W, layers, hidden_act, output_act):
    """Forward pass pour évaluation"""
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

# Tests (exactement comme dans cas_de_test_pmc.py)
tests_passed = 0
total_tests = 11

print("\n" + "="*60)
print("EXÉCUTION DES 11 TESTS")
print("="*60)

# Test 1: Linear Simple
print("\n1. Classification – Linear Simple")
try:
    X1 = np.array([[1,1],[2,3],[3,3]], dtype=np.float64)
    Y1 = np.array([1,-1,-1], dtype=np.float64)
    W, layers = train_pmc(X1, Y1, [10,10], 0, 0, lr=0.01, n_iters=1000)
    pred = pmc_forward(X1, W, layers, 0, 0)
    acc = np.mean((pred > 0.5) == (Y1 > 0))
    print(f"   Accuracy: {acc:.1%} {'PASS' if acc >= 0.5 else 'FAIL'}")
    if acc >= 0.5: tests_passed += 1
except Exception as e:
    print(f"   ERREUR: {e}")

# Test 2: Linear Multiple
print("\n2. Classification – Linear Multiple")
try:
    np.random.seed(0)
    X2 = np.vstack([
        np.random.random((50,2))*0.9 + [1,1],
        np.random.random((50,2))*0.9 + [2,2]
    ]).astype(np.float64)
    Y2 = np.concatenate([np.ones(50), -np.ones(50)]).astype(np.float64)
    W, layers = train_pmc(X2, Y2, [10,10], 0, 0, lr=0.01, n_iters=1000)
    pred = pmc_forward(X2, W, layers, 0, 0)
    acc = np.mean((pred > 0.5) == (Y2 > 0))
    print(f"   Accuracy: {acc:.1%} {'PASS' if acc >= 0.5 else 'FAIL'}")
    if acc >= 0.5: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 3: XOR
print("\n3. Classification – XOR")
try:
    X3 = np.array([[1,0],[0,1],[0,0],[1,1]], dtype=np.float64)
    Y3 = np.array([1,1,-1,-1], dtype=np.float64)
    W, layers = train_pmc(X3, Y3, [10,10], 0, 0, lr=0.05, n_iters=3000)
    pred = pmc_forward(X3, W, layers, 0, 0)
    acc = np.mean((pred > 0.5) == (Y3 > 0))
    print(f"   Accuracy: {acc:.1%} {'PASS' if acc >= 0.5 else 'FAIL'}")
    if acc >= 0.5: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 4: Cross
print("\n4. Classification – Cross")
try:
    np.random.seed(42)
    X4 = np.random.uniform(-1,1,(500,2)).astype(np.float64)
    Y4 = np.where((np.abs(X4[:,0])<=0.3)|(np.abs(X4[:,1])<=0.3), 1, -1).astype(np.float64)
    W, layers = train_pmc(X4, Y4, [10,10], 0, 0, lr=0.01, n_iters=1000)
    pred = pmc_forward(X4, W, layers, 0, 0)
    acc = np.mean((pred > 0.5) == (Y4 > 0))
    print(f"   Accuracy: {acc:.1%} {'PASS' if acc >= 0.5 else 'FAIL'}")
    if acc >= 0.5: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 5: Multi Linear 3 classes
print("\n5. Classification – Multi Linear 3 classes")
try:
    np.random.seed(123)
    X5 = np.random.uniform(-1,1,(500,2)).astype(np.float64)
    def label5(p):
        x,y = p
        if -x-y-0.5>0 and y<0 and x-y-0.5<0: return [1,0,0]
        if -x-y-0.5<0 and y>0 and x-y-0.5<0: return [0,1,0]
        if -x-y-0.5<0 and y<0 and x-y-0.5>0: return [0,0,1]
        return None
    labels5 = [label5(p) for p in X5]
    mask5 = [lab is not None for lab in labels5]
    X5_f, Y5 = X5[mask5], np.array([lab for lab in labels5 if lab is not None], dtype=np.float64)
    W, layers = train_pmc(X5_f, Y5, [10,10], 0, 1, lr=0.01, n_iters=1000)
    pred = pmc_forward(X5_f, W, layers, 0, 1)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(Y5, axis=1))
    print(f"   Accuracy: {acc:.1%} {'PASS' if acc >= 0.5 else 'FAIL'}")
    if acc >= 0.5: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 6: Multi Cross
print("\n6. Classification – Multi Cross")
try:
    np.random.seed(456)
    X6 = np.random.uniform(-1,1,(200,2)).astype(np.float64)
    def label6(p):
        x,y = p
        if abs(x%0.5)<=0.25 and abs(y%0.5)>0.25: return [1,0,0]
        if abs(x%0.5)>0.25 and abs(y%0.5)<=0.25: return [0,1,0]
        return [0,0,1]
    Y6 = np.array([label6(p) for p in X6], dtype=np.float64)
    W, layers = train_pmc(X6, Y6, [20,20], 1, 1, lr=0.01, n_iters=1500)
    pred = pmc_forward(X6, W, layers, 1, 1)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(Y6, axis=1))
    print(f"   Accuracy: {acc:.1%} {'PASS' if acc >= 0.5 else 'FAIL'}")
    if acc >= 0.5: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 7: Régression Linear Simple 2D
print("\n7. Régression – Linear Simple 2D")
try:
    X7 = np.array([[1],[2]], dtype=np.float64)
    Y7 = np.array([2,3], dtype=np.float64)
    W, layers = train_pmc(X7, Y7, [10,10], 0, 2, lr=0.01, n_iters=1000)
    pred = pmc_forward(X7, W, layers, 0, 2)
    mse = np.mean((pred.ravel() - Y7) ** 2)
    print(f"   MSE: {mse:.3f} {'PASS' if mse < 5.0 else 'FAIL'}")
    if mse < 5.0: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 8: Régression Non Linéaire Simple 2D
print("\n8. Régression – Non Linéaire Simple 2D")
try:
    X8 = np.array([[1],[2],[3]], dtype=np.float64)
    Y8 = np.array([2,3,2.5], dtype=np.float64)
    W, layers = train_pmc(X8, Y8, [10,10], 0, 2, lr=0.01, n_iters=1000)
    pred = pmc_forward(X8, W, layers, 0, 2)
    mse = np.mean((pred.ravel() - Y8) ** 2)
    print(f"   MSE: {mse:.3f} {'PASS' if mse < 5.0 else 'FAIL'}")
    if mse < 5.0: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 9: Régression Linear Simple 3D
print("\n9. Régression – Linear Simple 3D")
try:
    X9 = np.array([[1,1],[2,2],[3,1]], dtype=np.float64)
    Y9 = np.array([2,3,2.5], dtype=np.float64)
    W, layers = train_pmc(X9, Y9, [10,10], 0, 2, lr=0.01, n_iters=1000)
    pred = pmc_forward(X9, W, layers, 0, 2)
    mse = np.mean((pred.ravel() - Y9) ** 2)
    print(f"   MSE: {mse:.3f} {'PASS' if mse < 5.0 else 'FAIL'}")
    if mse < 5.0: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 10: Régression Linear Tricky 3D
print("\n10. Régression – Linear Tricky 3D")
try:
    X10 = np.array([[1,1],[2,2],[3,3]], dtype=np.float64)
    Y10 = np.array([1,2,3], dtype=np.float64)
    W, layers = train_pmc(X10, Y10, [10,10], 0, 2, lr=0.01, n_iters=1000)
    pred = pmc_forward(X10, W, layers, 0, 2)
    mse = np.mean((pred.ravel() - Y10) ** 2)
    print(f"   MSE: {mse:.3f} {'PASS' if mse < 5.0 else 'FAIL'}")
    if mse < 5.0: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Test 11: Régression Non Linéaire Simple 3D
print("\n11. Régression – Non Linéaire Simple 3D")
try:
    X11 = np.array([[1,0],[0,1],[1,1],[0,0]], dtype=np.float64)
    Y11 = np.array([2,1,-2,-1], dtype=np.float64)
    W, layers = train_pmc(X11, Y11, [10,10], 0, 2, lr=0.01, n_iters=2000)
    pred = pmc_forward(X11, W, layers, 0, 2)
    mse = np.mean((pred.ravel() - Y11) ** 2)
    print(f"   MSE: {mse:.3f} {'PASS' if mse < 5.0 else 'FAIL'}")
    if mse < 5.0: tests_passed += 1
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

# Résultats finaux
print("\n" + "="*60)
print("RÉSULTATS FINAUX")
print("="*60)
print(f"Tests réussis: {tests_passed}/{total_tests}")
print(f"Taux de réussite: {100*tests_passed/total_tests:.1f}%")

if tests_passed >= total_tests * 0.5:
    print(" PMC VALIDÉ - Plus de 50% de réussite!")
    status = "SUCCÈS"
else:
    print("FAIL PMC À AMÉLIORER - Moins de 50% de réussite")
    status = "ÉCHEC"

print("="*60)
print(f"STATUS FINAL: {status}")
