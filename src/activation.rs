use crate::matrix::Matrix;

pub fn relu(input: f64) -> f64 {
    if input < 0.0 {
        0.0
    } else {
        input
    }
}

pub fn softmax(input: &Matrix) -> Matrix {
    let mut input_sub_max: Matrix = Matrix::init_zero(input.height, input.width);

    for r in 0..input.height {
        let max: &f64 = input.data[r].iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        for c in 0..input.width {
            input_sub_max.data[r][c] = input.data[r][c] - max;
        }
    }

    let input_exp: Matrix = input_sub_max.exp();
    let mut output: Matrix = Matrix::init_zero(input.height, input.width);

    for r in 0..input.height {
        let sum: f64 = input_exp.data[r].iter().sum();
        for c in 0..input.width {
            output.data[r][c] = input_exp.data[r][c] / sum;
        }
    }

    output
}
