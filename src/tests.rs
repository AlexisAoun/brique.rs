#[cfg(test)]
mod tests {
    use crate::{layers::Layer, model::Model, parse_test_csv::parse_test_csv, Matrix};

    #[test]
    fn end_to_end_model_test() {
        let number_of_layers = 3;
        let input_weights: Vec<Matrix> = parse_test_csv("test_input_weights.csv".to_string());
        let test_data: Vec<Matrix> = parse_test_csv("test_data.csv".to_string());
        let expected_params: Vec<Matrix> = parse_test_csv("expected_params.csv".to_string());

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

        let layer1 = Layer::init_test(2, 3, true, input_weights[0].clone());
        let layer2 = Layer::init_test(3, 3, true, input_weights[1].clone());
        let layer3 = Layer::init_test(3, 3, false, input_weights[2].clone());

        let mut model = Model {
            layers: vec![layer1, layer2, layer3],
            lambda: 0.001,
            learning_step: 0.1,
            layers_debug: Vec::new(),
            input: Matrix::new(2, 2),
            input_label: Matrix::new(2, 2),
            itermediate_evaluation_results: Vec::new(),
            softmax_output: Matrix::new(2, 2),
            d_zs: Vec::new(),
            d_ws: Vec::new(),
            d_bs: Vec::new(),
            d_score: Matrix::new(2, 3),
            loss: 0.0,
            reg_loss: 0.0,
            data_loss: 0.0,
        };

        let network_history = model.train(&test_data[0], &test_data[1], 6, 5, 0, true);

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
                    "Gradient of the weights in iteration {}, layer {}, incorrect values",
                    index + 1,
                    i + 1
                )
            });

            // gradient of the biases
            model.d_bs.iter().enumerate().for_each(|(i, m)| {
                assert!(
                    m.is_equal(&expected_params[(index * 21) + (i * 3) + 13], precision),
                    "Gradient of the biases in iteration {}, layer {}, incorrect values",
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
                            "Gradient of the hidden layer score in iteration {}, layer {}, incorrect values",
                            index + 1,
                            i + 1
                        )
                    }
                });

            let loss_matrix = Matrix::init(1,3, vec![model.data_loss, model.reg_loss, model.loss]);
            assert!(loss_matrix.is_equal(&loss_matrix, precision), "Loss in iteration {}, incorrect values", index + 1);

            index += 1;
        }
    }

    fn get_rand_matrix_1() -> Matrix {
        let mut matrix_random_1: Matrix = Matrix::new(4, 2);
        matrix_random_1.data[0][0] = 2.0;
        matrix_random_1.data[0][1] = 1.0;
        matrix_random_1.data[1][0] = 4.0;
        matrix_random_1.data[1][1] = 5.6;
        matrix_random_1.data[2][0] = 23.0;
        matrix_random_1.data[2][1] = -0.4;
        matrix_random_1.data[3][0] = 0.0;
        matrix_random_1.data[3][1] = 3.0;

        matrix_random_1
    }

    fn get_rand_matrix_2() -> Matrix {
        let mut matrix_random_2: Matrix = Matrix::new(2, 3);
        matrix_random_2.data[0][0] = 2.0;
        matrix_random_2.data[0][1] = -0.69;
        matrix_random_2.data[0][2] = 6.5;
        matrix_random_2.data[1][0] = -1.0;
        matrix_random_2.data[1][1] = 1.0;
        matrix_random_2.data[1][2] = 0.5;

        matrix_random_2
    }

    fn get_rand_matrix_3() -> Matrix {
        let mut matrix_random_3: Matrix = Matrix::new(1, 3);
        matrix_random_3.data[0][0] = 8.0;
        matrix_random_3.data[0][1] = -5.21;
        matrix_random_3.data[0][2] = 0.00;

        matrix_random_3
    }

    #[test]
    fn test_dot() {
        let matrix_random_1 = get_rand_matrix_1();
        let matrix_random_2 = get_rand_matrix_2();

        let output = matrix_random_1.dot(&matrix_random_2);

        let mut expected_output: Matrix = Matrix::new(4, 3);
        expected_output.data[0][0] = 3.0;
        expected_output.data[0][1] = -0.3799999999999999;
        expected_output.data[0][2] = 13.5;
        expected_output.data[1][0] = 2.4000000000000004;
        expected_output.data[1][1] = 2.84;
        expected_output.data[1][2] = 28.8;
        expected_output.data[2][0] = 46.4;
        expected_output.data[2][1] = -16.27;
        expected_output.data[2][2] = 149.3;
        expected_output.data[3][0] = -3.0;
        expected_output.data[3][1] = 3.0;
        expected_output.data[3][2] = 1.5;

        assert!(output.is_equal(&expected_output, 10));
    }

    #[test]
    fn test_add_row() {
        let matrix_random_2 = get_rand_matrix_2();
        let matrix_random_3 = get_rand_matrix_3();

        let output = matrix_random_2.add_value_to_all_rows(&matrix_random_3);
        let mut expected_output: Matrix =
            Matrix::new(matrix_random_2.height, matrix_random_2.width);

        expected_output.data[0][0] = 10.0;
        expected_output.data[0][1] = -5.9;
        expected_output.data[0][2] = 6.5;
        expected_output.data[1][0] = 7.0;
        expected_output.data[1][1] = -4.21;
        expected_output.data[1][2] = 0.5;

        assert!(output.is_equal(&expected_output, 10));
    }
}
