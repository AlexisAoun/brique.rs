use crate::activation::*;
use crate::matrix::*;

// note : we have directly the transpose of weights (hence the _t) to optimize computation
#[derive(Clone)]
pub struct Layer {
    pub weights_t: Matrix,
    pub biases: Matrix,
    pub activation: bool,
    pub output: Matrix,
    pub d_output_next_layer: Matrix
}

impl Layer {
    pub fn init(input_size: u32, size: u32, activation: bool) -> Layer {
        Layer {
            weights_t: Matrix::init_rand(input_size.try_into().unwrap(), size.try_into().unwrap()),
            biases: Matrix::init_zero(1, size.try_into().unwrap()),
            activation,
            output: Matrix::init_zero(0, 0),
            d_output_next_layer: Matrix::init_zero(0, 0),
        }
    }

    #[allow(dead_code)]
    pub fn init_test(size: u32, activation: bool, weights_t: Matrix) -> Layer {
        Layer {
            weights_t,
            biases: Matrix::init_zero(1, size.try_into().unwrap()),
            activation,
            output: Matrix::init_zero(0, 0),
            d_output_next_layer: Matrix::init_zero(0, 0),
        }
    }

    pub fn forward(&mut self, input: &Matrix, predict: bool) -> Matrix {
        let mut tmp_output = input.dot(&self.weights_t);
        tmp_output = tmp_output.add_1d_matrix_to_all_rows(&self.biases);

        if self.activation {
            tmp_output = self.activation(&tmp_output);
        }

        if !predict {
            self.output = tmp_output.clone();
        }

        tmp_output
    }

    //implementing ReLu for this project
    fn activation(&self, input: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::init_zero(input.height, input.width);
        for r in 0..input.height {
            for c in 0..input.width {
                output.set(relu(input.get(r, c)), r, c);
            }
        }
        output
    }

    pub fn update_weigths(&mut self, input: &Matrix, learning_step: f64) {
        self.weights_t = self
            .weights_t
            .add_two_matrices(&input.mult(learning_step * -1.0));
    }

    pub fn update_biases(&mut self, input: &Matrix, learning_step: f64) {
        self.biases = self
            .biases
            .add_two_matrices(&input.mult(learning_step * -1.0));
    }
}
