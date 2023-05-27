use crate::Matrix;

pub fn relu(input: f64) -> f64 {
    if input < 0.0 {
        0.0
    } else {
        input
    }
}

pub fn softmax(input: &Matrix) -> Matrix {
    let input_exp: Matrix = input.exp();
    let mut output: Matrix = Matrix::new(input.height, input.width);

    for r in 0..input.height {
        let sum: f64 = input_exp.data[r].iter().sum();
        for c in 0..input.width {
            output.data[r][c] = input_exp.data[r][c] / sum;
        } 
    }

    output
}
