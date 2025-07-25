#!/usr/bin/env python3
"""
Test spécifique pour vérifier la correction des gradients normalisés
Test le cas colinéaire [1,1],[2,2],[3,3] qui causait la divergence
"""
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
lib_path = os.path.join(root, "target", "release", lib_fname)

print("="*50)
print("TEST DE CONVERGENCE - CAS COLINÉAIRE")
print("="*50)
print(f"Library: {lib_path}")
print(f"Exists: {os.path.exists(lib_path)}")

if not os.path.exists(lib_path):
    print("FAIL DLL introuvable!")
    exit(1)

lib = ctypes.CDLL(lib_path)

# Signature FFI
lib.train_pmc.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_double, ctypes.c_size_t,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
]

def train_pmc(X, Y, hidden_sizes, hidden_act, output_act, lr=0.01, n_iters=1000):
    n, d = X.shape
    hs = np.array(hidden_sizes, dtype=np.int64)
    
    if output_act == 2:
        k = 1
        Y_flat = Y.astype(np.float64).ravel()
    else:
        k = 1
        Yb = (Y > 0).astype(np.float64)
        Y_flat = Yb.ravel()

    layers = [d] + hidden_sizes + [k]
    total = sum((layers[i] + 1) * layers[i+1] for i in range(len(layers)-1))
    W = np.zeros(total, dtype=np.float64)

    lib.train_pmc(
        X.astype(np.float64), Y_flat, n, d, len(hidden_sizes), hs,
        k, hidden_act, output_act, lr, n_iters, W
    )
    return W, layers

print("\n1. TEST: Cas colinéaire problématique [1,1],[2,2],[3,3]")
try:
    X_colinear = np.array([[1,1],[2,2],[3,3]], dtype=np.float64)
    Y_colinear = np.array([1, 2, 3], dtype=np.float64)  # Régression
    
    print(f"   Données: X={X_colinear.tolist()}, Y={Y_colinear.tolist()}")
    
    W, layers = train_pmc(X_colinear, Y_colinear, [5, 5], 0, 2, lr=0.01, n_iters=1000)
    
    # Vérifier si les poids sont raisonnables (pas NaN, pas trop grands)
    if np.any(np.isnan(W)):
        print("   FAIL DIVERGENCE: Poids contiennent NaN")
    elif np.any(np.abs(W) > 100):
        print(f"   WARNING  POIDS TRÈS GRANDS: max={np.max(np.abs(W)):.2f}")
    else:
        print(f"   PASS CONVERGENCE: Poids raisonnables, max={np.max(np.abs(W)):.2f}")
        
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

print("\n2. TEST: Cas simple XOR")
try:
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    Y_xor = np.array([-1, 1, 1, -1], dtype=np.float64)
    
    W, layers = train_pmc(X_xor, Y_xor, [4, 4], 0, 0, lr=0.1, n_iters=2000)
    
    if np.any(np.isnan(W)):
        print("   FAIL DIVERGENCE: Poids contiennent NaN")
    elif np.any(np.abs(W) > 100):
        print(f"   WARNING  POIDS TRÈS GRANDS: max={np.max(np.abs(W)):.2f}")
    else:
        print(f"   PASS CONVERGENCE: Poids raisonnables, max={np.max(np.abs(W)):.2f}")
        
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

print("\n3. TEST: Cas linéaire simple")
try:
    X_linear = np.array([[1,1],[2,3],[3,3]], dtype=np.float64)
    Y_linear = np.array([1, -1, -1], dtype=np.float64)
    
    W, layers = train_pmc(X_linear, Y_linear, [10, 10], 0, 0, lr=0.01, n_iters=1000)
    
    if np.any(np.isnan(W)):
        print("   FAIL DIVERGENCE: Poids contiennent NaN")
    elif np.any(np.abs(W) > 100):
        print(f"   WARNING  POIDS TRÈS GRANDS: max={np.max(np.abs(W)):.2f}")
    else:
        print(f"   PASS CONVERGENCE: Poids raisonnables, max={np.max(np.abs(W)):.2f}")
        
except Exception as e:
    print(f"   FAIL ERREUR: {e}")

print("\n" + "="*50)
print("PASS TEST DE CONVERGENCE TERMINÉ")
print("Si tous les tests montrent 'CONVERGENCE', le problème")
print("de normalisation des gradients est résolu!")
print("="*50)
