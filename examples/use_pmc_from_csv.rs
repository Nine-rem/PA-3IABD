use pa_3a_iabd2::pmc::MLP;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage : cargo run --example use_pmc_from_csv -- <chemin/vers/fichier.csv>");
        return;
    }

    let path = &args[1];
    let (x, y) = load_csv(path);

    let input_size = x[0].len();
    let hidden_size = input_size * 2;
    let output_size = 1;

    let mut model = MLP::new(&[input_size, hidden_size, output_size]);
    model.train(&x, &y, 0.5, 3000);

    println!("\n=== PMC dynamique sur {}", path);
    for (xi, yi) in x.iter().zip(y.iter()) {
        let pred = model.predict(xi)[0];
        println!("Entrée: {:?} | Réel: {:.1} | Prédit: {:.4}", xi, yi[0], pred);
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
