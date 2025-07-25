import os
import ctypes
import platform
import numpy as np

# Configuration basique 
CRATE_NAME = "ml_library"
if platform.system() == "Windows":
    lib_fname = f"{CRATE_NAME}.dll"
else:
    lib_fname = f"lib{CRATE_NAME}.so"

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "ml_library"))
lib_path = os.path.join(root, "target", "debug", lib_fname)

print(f"Chemin DLL: {lib_path}")
print(f"Existe: {os.path.exists(lib_path)}")

if not os.path.exists(lib_path):
    raise FileNotFoundError(f"DLL introuvable: {lib_path}")

lib = ctypes.CDLL(lib_path)

# Test XOR simple - le plus important pour valider le MLP
print("=== TEST XOR ===")
X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
Y_xor = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)  # XOR logique

print(f"X_xor:\n{X_xor}")
print(f"Y_xor: {Y_xor}")

# Le test XOR est critique - si ça marche, le PMC est bon
# Essayons de découvrir une fonction qui marche
functions_to_try = ["train_pmc", "pmc_train", "train_mlp", "_train_pmc"]

for func_name in functions_to_try:
    try:
        func = getattr(lib, func_name)
        print(f"V/ Trouvé: {func_name}")
        break
    except AttributeError:
        print(f"X {func_name} non trouvée")
else:
    print("X Aucune fonction d'entraînement trouvée!")
    print("Les fonctions doivent être correctement exportées.")
