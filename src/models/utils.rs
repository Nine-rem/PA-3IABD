/// Transpose une matrice (vec de vec)
pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

/// Multiplie deux matrices
pub fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let rows_b = b.len();
    let cols_b = b[0].len();

    assert_eq!(cols_a, rows_b, "Dimensions incompatibles pour multiplication");

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

/// Multiplie une matrice par un vecteur
pub fn matvecmul(matrix: &Vec<Vec<f64>>, vector: &Vec<f64>) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    assert_eq!(cols, vector.len(), "Dimensions incompatibles pour multiplication matrice-vecteur");

    let mut result = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += matrix[i][j] * vector[j];
        }
        result[i] = sum;
    }
    result
}

/// Calcule l'inverse d'une matrice carrée via Gauss-Jordan
pub fn inverse(matrix: &Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    assert_eq!(n, matrix[0].len(), "La matrice doit être carrée");

    let mut a = matrix.clone();
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }

    for i in 0..n {
        let mut pivot = a[i][i];
        if pivot.abs() < 1e-12 {
            let mut found = false;
            for r in (i + 1)..n {
                if a[r][i].abs() > 1e-12 {
                    a.swap(i, r);
                    inv.swap(i, r);
                    pivot = a[i][i];
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
        for j in 0..n {
            a[i][j] /= pivot;
            inv[i][j] /= pivot;
        }
        for r in 0..n {
            if r != i {
                let factor = a[r][i];
                for c in 0..n {
                    a[r][c] -= factor * a[i][c];
                    inv[r][c] -= factor * inv[i][c];
                }
            }
        }
    }
    Some(inv)
}

/// Calcule la distance euclidienne au carré entre deux vecteurs
pub fn squared_euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Calcule la moyenne élément par élément d’un vecteur de vecteurs
pub fn mean_vector(vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    let len = vectors[0].len();
    let mut sum = vec![0.0; len];
    for v in vectors {
        for i in 0..len {
            sum[i] += v[i];
        }
    }
    let n = vectors.len() as f64;
    for i in 0..len {
        sum[i] /= n;
    }
    sum
}
