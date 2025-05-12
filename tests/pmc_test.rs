use pa_3a_iabd2::pmc::MLP;

#[test]
fn pmc_should_show_separation_on_xor() {
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut model = MLP::new(&[2, 8, 1]);
    model.train(&x, &y, 1.0, 5000);

    for (i, (xi, yi)) in x.iter().zip(y.iter()).enumerate() {
        let pred = model.predict(xi)[0];
        let expected = yi[0];
        let err = (pred - expected).abs();

        println!("Test #{i} – input: {:?} → expected: {:.1}, predicted: {:.3}, error: {:.3}",
            xi, expected, pred, err);

        // On tolère jusqu'à 0.4 d'erreur pour ne pas échouer à cause d’un plateau
        assert!(
            err < 0.4,
            "Le PMC ne parvient pas à différencier l'entrée {:?} (attendu {:.1}, obtenu {:.3})",
            xi, expected, pred
        );
    }
}
