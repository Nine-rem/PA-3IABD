@echo off
echo Activation de l'environnement virtuel Python...

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] Environnement virtuel activé
) else (
    echo [ERROR] Environnement virtuel non trouvé
    echo Création d'un nouvel environnement virtuel...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [OK] Nouvel environnement virtuel créé et activé
)

echo.
echo Installation des dépendances...
pip install numpy matplotlib

echo.
echo [OK] Environnement prêt !
echo Pour lancer les tests:
echo   python test_pmc_validation.py
echo   python test_full_pmc_no_graphics.py
echo   python ml_library\tests\cas_de_test_pmc.py

cmd /k
