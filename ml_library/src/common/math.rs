
/// Produit scalaire entre deux vecteurs.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Norme L2 d'un vecteur.
pub fn norm(a: &[f64]) -> f64 {
    dot_product(a, a).sqrt()
}

/// Addition de deux vecteurs.
pub fn add_vectors(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Soustraction de deux vecteurs.
pub fn sub_vectors(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Multiplication d’un vecteur par un scalaire.
pub fn scalar_mul(a: &[f64], scalar: f64) -> Vec<f64> {
    a.iter().map(|x| x * scalar).collect()
}

/// Produit élément par élément (Hadamard).
pub fn hadamard_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Applique une fonction scalaire sur chaque élément du vecteur.
pub fn apply_function(vec: &[f64], f: fn(f64) -> f64) -> Vec<f64> {
    vec.iter().map(|&x| f(x)).collect()
}

/// Produit matrice x vecteur.
pub fn mat_vec_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot_product(row, vector)).collect()
}

/// Produit matrice × matrice.
pub fn mat_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    assert!(a[0].len() == b.len(), "Dimensions incompatible");
    let mut c = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            // calcul sans allocation intermédiaire :
            let mut sum = 0.0;
            for k in 0..b.len() {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

/// Matrice de Gram (φᵀ φ).
pub fn gram_matrix(phi: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = phi.len();
    let mut g = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            g[i][j] = dot_product(&phi[i], &phi[j]);
        }
    }
    g
}

