mod lib;
use lib::linear_model::LinearModel;

fn main() {
    println!("Start");

    let mut model = LinearModel::new(3);

    let X = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 3.0, 4.0],
        vec![3.0, 4.0, 5.0],
    ];

    let y = vec![ 7.0, 9.0, 10.0];

    model.fit(&X, &y, 0.001, 10000);

    println!("Poids après l'entraînement : {:?}", model.get_weights());
    println!("Biais après l'entraînement : {}", model.get_bias());
}


