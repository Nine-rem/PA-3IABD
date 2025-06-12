use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use iabd2_pa::models::linear_regression::LinearRegression;
use iabd2_pa::models::model_trait::Model;
use csv::ReaderBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    // Récupérer le chemin du fichier CSV depuis les arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <chemin_fichier_csv>", args[0]);
        std::process::exit(1);
    }
    let csv_path = &args[1];

    // Lire le CSV
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(csv_path)?;

    let mut x_data: Vec<Vec<f64>> = Vec::new();
    let mut y_data: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        // suppose colonnes "x" et "y", ou adapte l’index
        let x_val: f64 = record[0].parse()?;
        let y_val: f64 = record[1].parse()?;

        x_data.push(vec![x_val]);
        y_data.push(y_val);
    }

    // Créer et entraîner le modèle
    let mut model = LinearRegression::new();
    model.train(&x_data, &y_data);

    // Sauvegarder les poids dans weights.txt
    let mut file = File::create("weights.txt")?;
    let weights = model.get_weights();
    let weights_str = weights.iter()
        .map(|w| w.to_string())
        .collect::<Vec<String>>()
        .join(",");
    file.write_all(weights_str.as_bytes())?;

    println!("Poids sauvegardés dans weights.txt : {}", weights_str);

    Ok(())
}
