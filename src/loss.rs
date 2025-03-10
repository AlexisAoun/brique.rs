use crate::layers::*;
use crate::matrix::*;

pub fn one_hot_encoding(input: &Matrix, labels: &Matrix) -> Matrix {
    assert_eq!(
        input.height, labels.width,
        "Input height and labels width should be equal"
    );
    let mut output = Matrix::init_zero(1, input.height);
    for c in 0..input.height {
        let v = input.get(c, labels.get(0, c) as usize);
        output.set(v, 0, c);
    }

    output
}

pub fn cross_entropy(output: &Matrix, labels: &Matrix) -> f64 {
    let output_one_hot: Matrix = one_hot_encoding(&output, &labels);

    let mut loss: f64 = 0.0;
    for c in 0..output_one_hot.width {
        loss += -output_one_hot.get(0, c).ln();
    }

    let output_loss = loss / output_one_hot.width as f64;

    output_loss
}

pub fn l2_reg(layers: &Vec<Layer>, lambda: f64) -> f64 {
    let mut l2: f64 = 0.0;

    for layer in layers {
        l2 += 0.5 * lambda * (layer.weights_t.pow(2).sum());
    }

    l2
}
