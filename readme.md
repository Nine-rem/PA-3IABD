# Projet Annuel 2024-2025 : Classification de Véhicules

## 📚 Description

Ce projet a pour but de concevoir et d'implémenter plusieurs modèles de Machine Learning en Rust, sans utiliser de bibliothèques externes, afin de résoudre une problématique de classification : **distinguer différents types de véhicules** à partir d'images.

Le projet comporte plusieurs étapes principales :
- Implémentation de modèles linéaires, perceptron multicouche (PMC), RBFN et SVM.
- Constitution et préparation d'un dataset d'images de véhicules.
- Création d'une API serveur en Rust permettant d'exposer les modèles.
- Développement d'une application cliente pour interagir avec les modèles.
- Analyse critique des résultats obtenus.

## 🛠 Technologies utilisées

- **Rust** (langage principal)
- **Matplotlib** (via Python pour la visualisation des résultats)
- **Jupyter Notebook** (résultats interactifs)
- **Git** (versionnage)

## 📂 Structure du projet

Projet_Vehicules/
├── README.md
├── rapport.md
├── dataset/
│   ├── voiture/
│   ├── camion/
│   └── moto/
├── src/
│   ├── lib/
│   │   ├── linear_model.rs
│   │   ├── pmc.rs
│   │   └── utils.rs
│   ├── server/
│   │   └── main.rs
│   └── tests/
│       └── test_models.rs
├── notebooks/
│   └── resultats.ipynb
├── docs/
│   └── schema_architecture.png
└── Cargo.toml


## 🚀 Comment démarrer
Pré-requis
Rust installé

Cargo pour la gestion de projet Rust

Un éditeur adapté : Visual Studio Code avec extension Rust, Rust Rover, ou autre

(Plus tard) Python 3 pour l'analyse des résultats


##🧪 Instructions pour exécuter le projet:

Cloner le projet :
git clone https://github.com/Nine-rem/PA-3IABD
cd Projet_Vehicules

Compiler le projet :
cargo build

Lancer les tests :
cargo test

Lancer le serveur (plus tard)
Une fois l’API serveur mise en place :
cargo run --bin server



✍️ Auteurs :
PALMIER Robin 
SENTHILNATHAN Kirtika
TE Mathis

Étudiants en Intelligence Artificielle et Big Data - ESGI
