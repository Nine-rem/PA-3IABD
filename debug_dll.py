import os
import ctypes
import platform

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
print(f"Library path: {lib_path}")
print(f"Exists: {os.path.exists(lib_path)}")

if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Bibliothèque introuvable : {lib_path}")

lib = ctypes.CDLL(lib_path)

# Liste toutes les fonctions disponibles 
print("\n[DEBUG] Fonctions disponibles dans la DLL:")
try:
    # Tenter d'accéder à des fonctions connues
    functions_to_test = [
        "train_pmc",
        "pmc_fit", 
        "pmc_predict",
        "pmc_save",
        "pmc_load"
    ]
    
    for func_name in functions_to_test:
        try:
            func = getattr(lib, func_name)
            print(f"v/ {func_name}: trouvée")
        except AttributeError:
            print(f"X {func_name}: non trouvée")
            
    # Test si d'autres fonctions sont disponibles
    print("\n[DEBUG] Test du module PMC:")
    print(f"Module lib object: {lib}")
    
except Exception as e:
    print(f"Erreur lors du test: {e}")
