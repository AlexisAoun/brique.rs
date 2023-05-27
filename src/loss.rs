use crate::{Matrix, activation::softmax};

pub fn one_hot_encoding(input: &Matrix, labels: &Matrix) -> Matrix {
    assert_eq!(input.height, labels.width, "Input height and labels width should be equal");
    let mut output = Matrix::new(1, input.height);
    for c in 0..input.height {
        output.data[0][c] = input.data[c][labels.data[0][c] as usize];
    }

    output
}

// implementing cross-entropy and L2 regulariztion
pub fn compute_loss(output: &Matrix, labels: &Matrix, lambda: f64) -> f64 {
    let output_one_hot: Matrix = one_hot_encoding(&output, &labels);
    let output_softmax: Matrix = softmax(&output_one_hot);

    let mut loss: f64 = 0.0;
    for c in 0..output_softmax.width {
        loss += -output_softmax.data[0][c].ln();
    }


    loss

}
