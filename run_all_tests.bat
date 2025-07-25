@echo off
echo ================================================================================
echo EXECUTION COMPLETE DES TESTS PMC
echo ================================================================================

cd /d "c:\Users\palmi\Documents\Code\ESGI\Projet Annuel\PA-3IABD"

echo.
echo [1/6] Verification de la DLL...
if exist "ml_library\target\release\ml_library.dll" (
    echo ✓ DLL trouvee: ml_library.dll
) else (
    echo ✗ DLL manquante - Compilation necessaire
    echo Compilation en cours...
    cd ml_library
    cargo build --release
    cd ..
)

echo.
echo [2/6] Test de debug DLL...
python debug_dll.py

echo.
echo [3/6] Test de convergence...
python test_convergence.py

echo.
echo [4/6] Tests de validation PMC...
python test_pmc_validation.py

echo.
echo [5/6] Test complet sans graphiques...
python test_full_pmc_no_graphics.py

echo.
echo [6/6] Tests unitaires Rust...
cd ml_library
cargo test --no-run >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Tests Rust compilent correctement
) else (
    echo ✗ Erreur compilation tests Rust
)
cd ..

echo.
echo ================================================================================
echo TOUS LES TESTS TERMINES
echo ================================================================================
echo.
echo Pour lancer les tests originaux avec visualisation:
echo   python ml_library\tests\cas_de_test_pmc.py
echo.
echo Rapport complet dans: CORRECTIONS_REPORT.txt
echo ================================================================================

pause
