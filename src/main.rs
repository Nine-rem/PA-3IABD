use pa_3a_iabd2::linear_model::LinearModel;
use pa_3a_iabd2::linear_regression::nonlinear_transform;

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage : cargo run -- <chemin/vers/fichier.csv>");
        return;
    }

    let path = &args[1];
    let (x, y) = load_csv(path);

    let x_t = nonlinear_transform(&x);
    let mut model = LinearModel::new(x_t[0].len());
    model.fit(&x_t, &y, 0.1, 2000);

    println!("\n=== Prédictions sur {}", path);
    for (xi, yi) in x_t.iter().zip(y.iter()) {
        let pred = model.predict(xi);
        println!("Réel: {:.1} | Prédit: {:.4}", yi, pred);
    }
}

fn load_csv(path: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
    let file = File::open(path).expect("Impossible d’ouvrir le fichier CSV");
    let reader = BufReader::new(file);
    let mut x = Vec::new();
    let mut y = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let parts: Vec<f64> = line.split(',').map(|v| v.trim().parse().unwrap()).collect();
        x.push(parts[..parts.len() - 1].to_vec());
        y.push(parts[parts.len() - 1]);
    }

    (x, y)
}
