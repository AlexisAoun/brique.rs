use crate::{activation::softmax, Layer, Matrix};
use crate::config::DEBUG;
use crate::log_matrix_into_csv;

pub fn one_hot_encoding(input: &Matrix, labels: &Matrix) -> Matrix {
    assert_eq!(
        input.height, labels.width,
        "Input height and labels width should be equal"
    );
    let mut output = Matrix::new(1, input.height);
    for c in 0..input.height {
        output.data[0][c] = input.data[c][labels.data[0][c] as usize];
    }

    output
}

pub fn cross_entropy(output: &Matrix, labels: &Matrix) -> f64 {
    let output_one_hot: Matrix = one_hot_encoding(&output, &labels);

    if DEBUG {
        log_matrix_into_csv("Begining cross_entropy, one hot encoding : ", &output_one_hot);
    }

    let mut loss: f64 = 0.0;
    for c in 0..output_one_hot.width {
        loss += -output_one_hot.data[0][c].ln();
    }

    let output_loss = loss / output_one_hot.width as f64;

    if DEBUG {
        println!("cross_entropy loss : {}", output_loss);
    }

    output_loss
}

pub fn l2_reg(layers: &Vec<Layer>, lambda: f64) -> f64 {
    let mut l2: f64 = 0.0;

    if DEBUG {
        println!("##### Begining L2 Reg #####");
    }

    for layer in layers {
        l2 += 0.5 * lambda * (layer.weights_t.pow(2).sum());
        if DEBUG {

            println!("tmp L2 : {}", l2);
        }
    }

    if DEBUG {
        println!("output L2 : {}", l2);
        println!("##### Ending L2 Reg #####");
    }

    l2
}
