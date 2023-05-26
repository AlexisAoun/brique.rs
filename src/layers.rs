use crate::matrix::*;
use crate::utils::*;

pub trait Layer {
    fn init(input_size: u32, size: u32) -> Self;
    fn forward(&self, input: &Matrix) -> Matrix;
}

// note : we have directly the transpose of weights (hence the _t) to optimize computation
pub struct ComputeLayer {
    pub weights_t: Matrix,
    pub biases: Matrix,
}

pub struct ActivationLayer;

impl Layer for ComputeLayer {
    fn init(input_size: u32, size: u32) -> Self {
        ComputeLayer {
            weights_t: Matrix::init_rand(input_size.try_into().unwrap(), size.try_into().unwrap()),
            biases: Matrix::new(1, size.try_into().unwrap()),
        }
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        let mut output: Matrix = input.dot(&self.weights_t);
        output = output.add_value_to_all_rows(&self.biases);

        output
    }
}

//implementing ReLu for this project
impl Layer for ActivationLayer {
    fn init(_input_size: u32, _size: u32) -> Self {
        ActivationLayer {}
    }

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
