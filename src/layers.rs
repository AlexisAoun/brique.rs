use crate::matrix::*;
use crate::utils::*;
use crate::activation::*;

pub trait Layer {
    fn forward(&self, input: &Matrix) -> Matrix;
}

// note : we have directly the transpose of weights (hence the _t) to optimize computation
pub struct ComputeLayer {
    pub weights_t: Matrix,
    pub biases: Matrix,
}

pub struct ActivationLayer;

impl Layer for ComputeLayer {
    // fn init(&mut self, input_size: u32, size: u32) {
    //     self = ComputeLayer {
    //         weights_t: Matrix::init_rand(input_size.try_into().unwrap(), size.try_into().unwrap()),
    //         biases: Matrix::new(1, size.try_into().unwrap()),
    //     };
    // }

    fn forward(&self, input: &Matrix) -> Matrix {
        let mut output: Matrix = input.dot(&self.weights_t);
        output = output.add_value_to_all_rows(&self.biases);

        output
    }
}

//implementing ReLu for this project
impl Layer for ActivationLayer {
    fn forward(&self, input: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::new(input.height, input.width);
        for r in 0..input.height {
            for c in 0..input.width {
                output.data[r][c] = relu(input.data[r][c]);
            }
        }
        output
    }
}

// TODO init fun that takes all the layers. should put it in model ? 
