minimum 20 pages, présentant l’étude des performances de tous les algorithmes et modèles sur la problé matique choisie.

Prob : faire un model qui differencie et classes des images de divers vehicules(Voiture, camion, moto, vélo).
Repo Git : https://github.com/Nine-rem/PA-3IABD/tree/rendu2-Robin
Dataset : pour construire le dataset, il nous faut une grande quantité de photos prise avec des appareils différent.
    * Les photos doivent être bien centrée et si possible pas d'autres types de véhicules présent sur la photo.
    * Le véhicule doit être bien reconnaissable (pas de photo de véhicule à 100m par exemple).
    * Bien mettre les images dans la bonne catégorie (pour trier les photos plus rapidement).

Notes:
    Dataset:
        il nous faut trouver un moyen de standardiser les images.

    LinearModel: 
     les situations où ton modèle linéaire se trompe visiblement :
    - Soit les erreurs sont trop grandes (en régression),
    - Soit le modèle classe mal (en classification),
    - Soit les résultats ne collent pas à la forme des données (ex :
      données non-linéaires).


Performances:
    Algorithmes :
    Modèles:




11 mai 2025
## Objectif de l’étape

L’objectif de cette étape est d’implémenter et tester les premiers modèles de Machine Learning sans utiliser de bibliothèques externe. Les modèles devaient être testés sur des jeux de données simples ainsi qu’une portion du dataset final.

- Implémentation d’un modèle linéaire et d’un PMC
- Mise en place de transformations non linéaires
- Tests pour valider les apprentissages

## Modèles implémentés

### 1. Modèle Linéaire
- Implémenté avec fonction d’activation `sigmoid` pour la classification binaire.
- Optimisation par descente de gradient.
- Testé d’abord sur données jouets puis sur fichier `dataset.csv`.

### 2. Transformation Non Linéaire
- Pour améliorer les performances du modèle linéaire sur des données non linéairement séparables.
- Transformation appliquée : `x → [x, x², sin(x)]`.

### 3. Perceptron Multi-Couches (PMC)
- Implémentation d’un MLP avec propagation avant et rétropropagation.
- Architecture flexible : `[input_size, hidden, output]`
- Fonction d’activation : `sigmoid`
- Testé sur données XOR (`pmc_dataset.csv`).

 
## 🔍 Tests et résultats

### 🔸 Modèle Linéaire
Test : séparation de deux classes simulées  
Résultat : prédictions correctes avec une transformation `x, x², sin(x)`

### 🔸 PMC (Perceptron Multi-Couche)
Test : apprentissage du XOR  
Résultat : convergence stable (prédictions correctes dans tolérance < 0.4)

### 🔸 Transformation non linéaire
Test : vérification que les colonnes sont bien transformées (x, x², sin(x))  
Résultat : exactitude numérique vérifiée




### Analyse
Le modèle linéaire montre des limites claires sur les données XOR ou non séparables sans transformation.

L’ajout d’une transformation non linéaire améliore significativement les résultats.

Le PMC surpasse facilement le modèle linéaire dès qu’il s’agit de capter des relations complexes (non linéaires).

L’architecture modulaire (lib séparée + tests en examples/).