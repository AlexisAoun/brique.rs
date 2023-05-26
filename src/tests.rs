#[cfg(test)]
mod tests {
    use crate::Matrix;
    #[test]
    fn test_dot() {
        let mut matrix_random_1: Matrix = Matrix::new(4,2);
        matrix_random_1.data[0][0] = 2.0;
        matrix_random_1.data[0][1] = 1.0;
        matrix_random_1.data[1][0] = 4.0;
        matrix_random_1.data[1][1] = 5.6;
        matrix_random_1.data[2][0] = 23.0;
        matrix_random_1.data[2][1] = -0.4;
        matrix_random_1.data[3][0] = 0.0;
        matrix_random_1.data[3][1] = 3.0;
        let mut matrix_random_2: Matrix = Matrix::new(2,3);
        matrix_random_2.data[0][0] = 2.0;
        matrix_random_2.data[0][1] = -0.69;
        matrix_random_2.data[0][2] = 6.5;
        matrix_random_2.data[1][0] = -1.0;
        matrix_random_2.data[1][1] = 1.0;
        matrix_random_2.data[1][2] = 0.5;

        let output = matrix_random_1.dot(&matrix_random_2);

        let mut expected_output: Matrix = Matrix::new(4,3);
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

        assert!(output.isEqual(&expected_output));
    }

}

