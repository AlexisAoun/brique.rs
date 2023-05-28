use crate::matrix::*;
use crate::utils::*;
use crate::activation::*;

// note : we have directly the transpose of weights (hence the _t) to optimize computation
pub struct Layer {
    pub weights_t: Matrix,
    pub biases: Matrix,
    pub activation: bool
}

impl Layer {
    pub fn init(input_size: u32, size: u32, activation: bool) -> Layer {
        Layer {
            weights_t: Matrix::init_rand(input_size.try_into().unwrap(), size.try_into().unwrap()),
            biases: Matrix::new(1, size.try_into().unwrap()),
            activation
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let mut output: Matrix = input.dot(&self.weights_t);
        output = output.add_value_to_all_rows(&self.biases);

        if self.activation {
            self.activation(&output)
        } else {
            output
        }
    }

    //implementing ReLu for this project
    fn activation(&self, input: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::new(input.height, input.width);
        for r in 0..input.height {
            for c in 0..input.width {
                output.data[r][c] = relu(input.data[r][c]);
            }
        }
        output
    }
}

