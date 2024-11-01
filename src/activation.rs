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
        let max: f64 = *input
            .get_row(r)
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        for c in 0..input.width {
            let v = input.get(r, c) - max;
            input_sub_max.set(v, r, c);
        }
    }

    let input_exp: Matrix = input_sub_max.exp();
    let mut output: Matrix = Matrix::init_zero(input.height, input.width);

    for r in 0..input.height {
        let sum: f64 = input_exp.get_row(r).iter().sum();
        for c in 0..input.width {
            let v = input_exp.get(r, c) / sum;
            output.set(v, r, c);
        }
    }

    output
}
