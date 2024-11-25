#[cfg(test)]
mod tests {
    use brique::{layers::Layer, matrix::*, model::Model, parse_test_csv::parse_test_csv};

    #[test]
    fn end_to_end_model_test() {
        let number_of_layers = 3;
        let input_weights: Vec<Matrix> =
            parse_test_csv("tests/test_data/test_input_weights.csv".to_string());
        let test_data: Vec<Matrix> = parse_test_csv("tests/test_data/test_data.csv".to_string());
        let expected_params: Vec<Matrix> =
            parse_test_csv("tests/test_data/expected_params.csv".to_string());

        assert_eq!(
            input_weights.len(),
            number_of_layers,
            "The input weight csv doesn't have the expected number of matrices {}",
            number_of_layers
        );
        assert_eq!(
            test_data.len(),
            2,
            "The test data csv doesn't have the expected number of matrices {}",
            number_of_layers
        );

        let layer1 = Layer::init_test(3, true, input_weights[0].clone());
        let layer2 = Layer::init_test(3, true, input_weights[1].clone());
        let layer3 = Layer::init_test(3, false, input_weights[2].clone());

        let mut model = Model {
            layers: vec![layer1, layer2, layer3],
            lambda: 0.001,
            learning_step: 0.1,
            layers_debug: Vec::new(),
            input: Matrix::init_zero(2, 2),
            input_label: Matrix::init_zero(2, 2),
            itermediate_evaluation_results: Vec::new(),
            softmax_output: Matrix::init_zero(2, 2),
            d_zs: Vec::new(),
            d_ws: Vec::new(),
            d_bs: Vec::new(),
            d_score: Matrix::init_zero(2, 3),
            loss: 0.0,
            reg_loss: 0.0,
            data_loss: 0.0,
        };

        let network_history = model.train(&test_data[0], &test_data[1], 6, 5, 0, true, true);

        let models: Vec<Model> = network_history.unwrap();

        let mut index: usize = 0;
        let precision: i32 = 10;

        for model in models {
            // weights and biases
            model.layers_debug.iter().enumerate().for_each(|(i, l)| {
                assert!(
                    l.weights_t
                        .is_equal(&expected_params[(index * 21) + (i * 2)], precision),
                    "Weights in iteration {}, layer {}, incorrect values",
                    index + 1,
                    i + 1
                );
                assert!(
                    l.biases
                        .is_equal(&expected_params[(index * 21) + (i * 2) + 1], precision),
                    "Biases in iteration {}, layer {}, incorrect values",
                    index + 1,
                    i + 1
                );
            });

            // intermediate layer results
            model
                .itermediate_evaluation_results
                .iter()
                .enumerate()
                .for_each(|(i, m)| {
                    assert!(
                        m.is_equal(&expected_params[(index * 21) + i + 6], precision),
                        "Intermediate evaluation result in iteration {}, layer {}, incorrect values",
                        index + 1,
                        i + 1
                    )
                });

            // softmax result
            assert!(
                model
                    .softmax_output
                    .is_equal(&expected_params[(index * 21) + 10], precision),
                "softmax output in iteration {}, incorrect values",
                index + 1,
            );

            // gradient of the loss
            assert!(
                model
                    .d_score
                    .is_equal(&expected_params[(index * 21) + 11], precision),
                "gradient of loss d_score in iteration {}, incorrect values",
                index + 1,
            );

            // gradient of the weights
            model.d_ws.iter().enumerate().for_each(|(i, m)| {
                assert!(
                    m.is_equal(&expected_params[(index * 21) + (i * 3) + 12], precision),
                    "Gradient of the weights in iteration {}, index {}, incorrect values",
                    index + 1,
                    i + 1
                )
            });

            // gradient of the biases
            model.d_bs.iter().enumerate().for_each(|(i, m)| {
                assert!(
                    m.is_equal(&expected_params[(index * 21) + (i * 3) + 13], precision),
                    "Gradient of the biases in iteration {}, index {}, incorrect values",
                    index + 1,
                    i + 1
                )
            });

            // gradient of the hidden layers scores
            model
                .d_zs
                .iter()
                .enumerate()
                .for_each(|(i, m)| {
                    if i > 0 {
                        assert!(
                            m.is_equal(&expected_params[(index * 21) + ((i-1)*3) + 14], precision),
                            "Gradient of the hidden layer score in iteration {}, index {}, incorrect values",
                            index + 1,
                            i + 1
                        )
                    }
                });

            let loss_matrix = Matrix::init(1, 3, vec![model.data_loss, model.reg_loss, model.loss]);
            loss_matrix.display();
            expected_params[((index + 1) * 21) - 1].display();
            assert!(
                loss_matrix.is_equal(&expected_params[((index + 1) * 21) - 1], precision),
                "Loss in iteration {}, incorrect values",
                index + 1
            );

            index += 1;
        }
    }
}
