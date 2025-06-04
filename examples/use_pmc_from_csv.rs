use pa_3a_iabd2::lib::pmc::MLP;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let (x, y) = load_csv("src/data/pmc_results.csv");

    let mut mlp = MLP::new(&[x[0].len(), 3, 1]);
    mlp.train(&x, &y, 0.1, 1000);

    println!("=== PMC Prédictions ===");
    for (xi, yi) in x.iter().zip(y.iter()) {
        let out = mlp.predict(xi);
        println!("Entrée: {:?} | Réel: {:.1} | Prédit: {:.4}", xi, yi[0], out[0]);
    }
}

fn load_csv(path: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let file = File::open(path).expect("Impossible d’ouvrir le fichier CSV");
    let reader = BufReader::new(file);
    let mut x = Vec::new();
    let mut y = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let parts: Vec<f64> = line.split(',').map(|v| v.trim().parse().unwrap()).collect();
        x.push(parts[..parts.len() - 1].to_vec());
        y.push(vec![parts[parts.len() - 1]]);
    }

    (x, y)
}
