pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn scalar_mul(a: &[f64], scalar: f64) -> Vec<f64> {
    a.iter().map(|x| x * scalar).collect()
}

pub fn vector_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
