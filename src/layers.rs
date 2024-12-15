use crate::activation::*;
use crate::matrix::*;

// note : we have directly the transpose of weights (hence the _t) to optimize computation
#[derive(Clone)]
pub struct Layer {
    pub weights_t: Matrix,
    pub biases: Matrix,
    pub activation: bool,
    pub output: Matrix,
}

impl Layer {
    pub fn init(input_size: u32, size: u32, activation: bool) -> Layer {
        Layer {
            weights_t: Matrix::init_rand(input_size.try_into().unwrap(), size.try_into().unwrap()),
            biases: Matrix::init_zero(1, size.try_into().unwrap()),
            activation,
            output: Matrix::init_zero(0, 0),
        }
    }

    #[allow(dead_code)]
    pub fn init_test(size: u32, activation: bool, weights_t: Matrix) -> Layer {
        Layer {
            weights_t,
            biases: Matrix::init_zero(1, size.try_into().unwrap()),
            activation,
            output: Matrix::init_zero(0, 0),
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

    // d_w(i) = input(i) * d_output(i)
    // can be rewritten
    // d_w(i) = output(i-1) * d_output(i) -> if i > 0
    // d_b(i) = sum of the rows of d_output(i)
    // d_output(i) = d_output(i+1) * d_w(i+1)
    pub fn backprop(
        &mut self,
        d_z: &Matrix,
        z_minus_1: &Matrix,
        previous_layer_activation: bool,
        lambda: f64,
        learning_step: f64,
        is_input_layer: bool,
        debug: bool,
        debug_array_d_weights: &mut Option<Vec<Matrix>>,
        debug_array_d_biaises: &mut Option<Vec<Matrix>>,
        debug_array_d_outputs: &mut Option<Vec<Matrix>>,
    ) -> Matrix {
        let d_w: Matrix = z_minus_1
            .t()
            .dot(d_z)
            .add_two_matrices(&self.weights_t.mult(lambda));
        let d_b: Matrix = d_z.sum_rows();

        if debug {
            debug_array_d_outputs
                .get_or_insert_with(|| Vec::new())
                .push(d_z.clone());
            debug_array_d_weights
                .get_or_insert_with(|| Vec::new())
                .push(d_w.clone());
            debug_array_d_biaises
                .get_or_insert_with(|| Vec::new())
                .push(d_b.clone());
        }

        let mut new_d_z = d_z.dot(&self.weights_t.t());
        if !is_input_layer {
            if previous_layer_activation {
                new_d_z.compute_d_relu_inplace(z_minus_1);
            }
        }

        self.update_weigths(&d_w, learning_step);
        self.update_biases(&d_b, learning_step);

        new_d_z
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
