use crate::matrix::Matrix;

pub fn relu(input: f64) -> f64 {
    if input < 0.0 {
        0.0
    } else {
        input
    }
}

pub fn softmax(input: &Matrix) -> Matrix {
    let mut max_per_row: Vec<f64> = vec![];

    for r in 0..input.height {
        let max: f64 = *input
            .get_row(r)
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        max_per_row.push(max);
    }

    let vec_output: Vec<f64> = (0..input.height * input.width)
        .map(|i| (input.get_1d(i) - max_per_row[i / input.width]).exp())
        .collect();

    let mut output = Matrix::init(input.height, input.width, vec_output);

    let mut sum_per_row: Vec<f64> = vec![];

    for r in 0..output.height {
        let sum: f64 = output.get_row(r).iter().sum();

        sum_per_row.push(sum);
    }

    output.data = output
        .data
        .iter()
        .enumerate()
        .map(|(i, v)| v / sum_per_row[i / output.width])
        .collect();

    output
}
