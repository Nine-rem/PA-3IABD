use rand::Rng;

pub struct Perceptron {
    pub weights: Vec<f64>,
    pub learning_rate: f64,
}

impl Perceptron { 
    
    pub fn new(n_features: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..=n_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Self { weights, learning_rate }
    }

    /// Prédit +1 ou -1
    pub fn predict(&self, input: &[f64]) -> f64 {
        // Vérifie qu’on a bien un poids de biais + un par feature
        assert_eq!(input.len() + 1, self.weights.len(),
            "input.len()+1 doit == weights.len()");
        let sum = self.weights[0] + self.weights[1..]
            .iter()
            .zip(input)
            .map(|(w, x)| w * x)
            .sum::<f64>();
        if sum >= 0.0 { 1.0 } else { -1.0 }
    }
}
