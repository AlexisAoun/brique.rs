use crate::matrix::*;
use crate::utils::*;

pub trait Layer {
    fn init(input_size: u32, size: u32) -> Self;
    fn forward(&self, input: &Matrix) -> Matrix;
}

pub struct ComputeLayer {
    pub weights: Matrix,
    pub biases: Matrix,
}

pub struct ActivationLayer;

impl Layer for ComputeLayer {
    fn init(input_size: u32, size: u32) -> Self {
        ComputeLayer {
            weights: Matrix::new(size.try_into().unwrap(), input_size.try_into().unwrap()),
            biases: Matrix::new(size.try_into().unwrap(), 1),
        }
        // TODO randomize weights
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        let mut output: Matrix = input.dot(&self.weights);
        output = output.add_value_to_all_columns(&self.biases);

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
