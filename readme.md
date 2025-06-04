# Projet Annuel 2024-2025 : Classification de Véhicules

## 📚 Description

Ce projet vise à implémenter plusieurs modèles de machine learning en Rust, sans bibliothèques externes, pour classifier des types de véhicules à partir de données simulées. Il comprend :

- Un modèle linéaire avec transformation non linéaire
- Un perceptron multicouche (PMC)
- Des tests unitaires
- Une structure modulaire et réutilisable

## 🛠 Technologies

- Rust (sans crate externe pour le ML)
- Tests unitaires avec `cargo test`
- Dataset au format `.csv`
- Exemples d'utilisation via `cargo run --example`

## 📂 Structure

```
PA-3A-IABD2/
├── src/
│   ├── main.rs
│   └── lib/
│       ├── mod.rs
│       ├── linear_model.rs
│       ├── linear_regression.rs
│       └── pmc.rs
├── examples/
│   ├── use_with_csv.rs
│   └── use_pmc_with_csv.rs
├── tests/
│   ├── linear_model_test.rs
│   ├── pmc_test.rs
│   └── nonlinear_transform_test.rs
├── data/
│   ├── dataset.csv
│   └── pmc_dataset.csv
```

## ▶️ Lancer le projet

```bash
# Compiler
cargo build

# Lancer un exemple
cargo run --example use_with_csv
cargo run --example use_pmc_from_csv -- src/data/pmc_dataset.csv


# Exécuter les tests
cargo test
```

## ✅ Tests disponibles

- `linear_model_test.rs` : vérifie que le modèle linéaire apprend à séparer deux classes avec une transformation.
- `pmc_test.rs` : valide que le PMC apprend le XOR de manière fiable avec tolérance.
- `nonlinear_transform_test.rs` : assure que la transformation applique bien `x`, `x²` et `sin(x)`.

## ✍️ Auteurs

Robin Palmier, Kirtika Senthilnathan, Mathis Te  
Étudiants en Intelligence Artificielle & Big Data – ESGI
