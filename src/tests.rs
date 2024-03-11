#[cfg(test)]
mod tests {
    use crate::{Matrix, parse_test_csv::parse_test_csv, model::Model, layers::Layer, config::TEST};

    #[test]
    fn end_to_end_model_test() {
        unsafe {
            TEST = true;
        }

        let number_of_layers = 3;
        let input_weights: Vec<Matrix> = parse_test_csv("test_input_weights.csv".to_string());

        assert_eq!(input_weights.len(), number_of_layers, "The input weight csv doesn't have the expected number of matrices {}", number_of_layers);

        let layer1 = Layer::init_test(2, 3, true, input_weights[0].clone());
        let layer2 = Layer::init_test(3, 3, true, input_weights[1].clone());
        let layer3 = Layer::init_test(3, 3, false, input_weights[2].clone());

        let mut model = Model {
            layers: vec![layer1, layer2, layer3],
            lambda: 0.001,
            learning_step: 1.0,
        };

        // TODO Choose how to get the weights and biases at every stages
        // i should get only the final weights and biases to simplify things a bit
        // maybe something in the same fashion as the debug env variable i tried


        // to get the states i have to make it return by the training func
        // make it return an option, null if test is false
        // if true an array of n layers multiplied by the number of iterations
        // giving me a full history of the weights and biases evolution


        unsafe {
            TEST = false;
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

        assert!(output.is_equal(&expected_output));
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

        assert!(output.is_equal(&expected_output));
    }
}
