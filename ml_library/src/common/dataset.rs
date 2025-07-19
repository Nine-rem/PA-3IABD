use std::fs::File;
use std::io::{BufRead, BufReader};

/// Charge un dataset CSV (sans header) en deux vecteurs : features (X) et labels (y).
/// Assumes que la dernière colonne est le label.
pub fn load_csv_dataset(path: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
    let file = File::open(path).expect("Impossible d'ouvrir le fichier CSV.");
    let reader = BufReader::new(file);

    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let values: Vec<f64> = line.split(',')
            .map(|s| s.trim().parse::<f64>().expect("Erreur de parsing"))
            .collect();

        let (features, label) = values.split_at(values.len() - 1);
        x_data.push(features.to_vec());
        y_data.push(label[0]);
    }

    (x_data, y_data)
}

/// Sépare le dataset en train/test selon un ratio (par ex. 0.8).
pub fn train_test_split<T: Clone>(data: &[T], labels: &[f64], ratio: f64) -> (Vec<T>, Vec<f64>, Vec<T>, Vec<f64>) {
    let train_size = (data.len() as f64 * ratio).round() as usize;

    let x_train = data[..train_size].to_vec();
    let y_train = labels[..train_size].to_vec();

    let x_test = data[train_size..].to_vec();
    let y_test = labels[train_size..].to_vec();

    (x_train, y_train, x_test, y_test)
}
