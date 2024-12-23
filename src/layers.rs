use crate::activation::*;
use crate::matrix::*;
use crate::optimizer::Optimizer;

const EPSILON: f64 = 10E-8;
// note : we have directly the transpose of weights (hence the _t)
// height -> number of inputs
// width -> number of neurons in the layer
#[derive(Clone)]
pub struct Layer {
    pub weights_t: Matrix,
    pub biases: Matrix,
    pub activation: bool,
    pub output: Matrix,

    // for adam optimizer
    pub first_moment_weight: Option<Matrix>,
    pub first_moment_biase: Option<Matrix>,
    pub second_moment_weight: Option<Matrix>,
    pub second_moment_biase: Option<Matrix>,
}

impl Layer {
    pub fn init(input_size: u32, size: u32, activation: bool) -> Layer {
        Layer {
            weights_t: Matrix::init_rand(input_size.try_into().unwrap(), size.try_into().unwrap()),
            biases: Matrix::init_zero(1, size.try_into().unwrap()),
            activation,
            output: Matrix::init_zero(0, 0),
            first_moment_weight: None,
            first_moment_biase: None,
            second_moment_weight: None,
            second_moment_biase: None,
        }
    }

    pub fn init_with_data(weights_t: Matrix, biases: Matrix, activation: bool) -> Layer {
        Layer {
            weights_t,
            biases,
            activation,
            output: Matrix::init_zero(0, 0),
            first_moment_weight: None,
            first_moment_biase: None,
            second_moment_weight: None,
            second_moment_biase: None,
        }
    }

    #[allow(dead_code)]
    pub fn init_test(size: u32, activation: bool, weights_t: Matrix) -> Layer {
        Layer {
            weights_t,
            biases: Matrix::init_zero(1, size.try_into().unwrap()),
            activation,
            output: Matrix::init_zero(0, 0),
            first_moment_weight: None,
            first_moment_biase: None,
            second_moment_weight: None,
            second_moment_biase: None,
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
        optimizer: &Optimizer,
        iteration: i32,
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

        self.update_weigths(d_w, optimizer, iteration);
        self.update_biases(d_b, optimizer, iteration);

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

    pub fn update_weigths(&mut self, input: Matrix, optimizer: &Optimizer, iteration: i32) {
        match optimizer {
            Optimizer::SGD { learning_step } => {
                self.weights_t = self
                    .weights_t
                    .add_two_matrices(&input.mult(*learning_step * -1.0));
            }

            Optimizer::Adam {
                learning_step,
                beta1,
                beta2,
            } => {
                let mut corrected_first_moment: Matrix =
                    self.compute_corrected_first_moment_weights(input.clone(), *beta1, iteration);
                let mut corrected_second_moment: Matrix =
                    self.compute_corrected_second_moment_weights(input, *beta2, iteration);

                // W(t+1) = W(t) - learning_step * (first_moment / (Sqrt(second_moment) + epsilon))
                corrected_second_moment.sqrt_inplace();
                corrected_second_moment.add_inplace(EPSILON);
                corrected_first_moment.div_two_matrices_inplace(&corrected_second_moment);

                self.weights_t = self
                    .weights_t
                    .add_two_matrices(&corrected_first_moment.mult(*learning_step * -1.0));
            }
        }
    }

    pub fn update_biases(&mut self, input: Matrix, optimizer: &Optimizer, iteration: i32) {
        match optimizer {
            Optimizer::SGD { learning_step } => {
                self.biases = self
                    .biases
                    .add_two_matrices(&input.mult(*learning_step * -1.0));
            }

            Optimizer::Adam {
                learning_step,
                beta1,
                beta2,
            } => {
                let mut corrected_first_moment: Matrix =
                    self.compute_corrected_first_moment_biases(input.clone(), *beta1, iteration);
                let mut corrected_second_moment: Matrix =
                    self.compute_corrected_second_moment_biases(input, *beta2, iteration);

                // B(t+1) = B(t) - learning_step * (first_moment / (Sqrt(second_moment) + epsilon))
                corrected_second_moment.sqrt_inplace();
                corrected_second_moment.add_inplace(EPSILON);
                corrected_first_moment.div_two_matrices_inplace(&corrected_second_moment);

                self.biases = self
                    .biases
                    .add_two_matrices(&corrected_first_moment.mult(*learning_step * -1.0));
            }
        }
    }

    fn compute_corrected_first_moment_weights(
        &mut self,
        mut input: Matrix,
        beta1: f64,
        iteration: i32,
    ) -> Matrix {
        // first_moment (t+1) = momentum(t) * beta1 + gradient(t+1) * (1 - beta1)
        input.mult_inplace(1.0 - beta1);
        self.first_moment_weight
            .get_or_insert(Matrix::init_zero(
                self.weights_t.height,
                self.weights_t.width,
            ))
            .mult_inplace(beta1);
        self.first_moment_weight
            .get_or_insert(Matrix::init_zero(
                self.weights_t.height,
                self.weights_t.width,
            ))
            .add_two_matrices_inplace(&input);

        match &self.first_moment_weight {
            Some(first_moment) => first_moment.div(1.0 - (beta1.powi(iteration))),
            None => panic!("Weight first_moment should be initalized at this point"),
        }
    }

    fn compute_corrected_first_moment_biases(
        &mut self,
        mut input: Matrix,
        beta1: f64,
        iteration: i32,
    ) -> Matrix {
        // first_moment (t+1) = momentum(t) * beta1 + gradient(t+1) * (1 - beta1)
        input.mult_inplace(1.0 - beta1);
        self.first_moment_biase
            .get_or_insert(Matrix::init_zero(self.biases.height, self.biases.width))
            .mult_inplace(beta1);
        self.first_moment_biase
            .get_or_insert(Matrix::init_zero(self.biases.height, self.biases.width))
            .add_two_matrices_inplace(&input);

        match &self.first_moment_biase {
            Some(first_moment) => first_moment.div(1.0 - (beta1.powi(iteration))),
            None => panic!("Biase first_moment should be initalized at this point"),
        }
    }

    fn compute_corrected_second_moment_weights(
        &mut self,
        mut input: Matrix,
        beta2: f64,
        iteration: i32,
    ) -> Matrix {
        // second_moment (t+1) = second_momentu(t) * beta2 + gradient(t+1)² * (1 - beta1)
        input.pow_inplace(2);
        input.mult_inplace(1.0 - beta2);
        self.second_moment_weight
            .get_or_insert(Matrix::init_zero(
                self.weights_t.height,
                self.weights_t.width,
            ))
            .mult_inplace(beta2);
        self.second_moment_weight
            .get_or_insert(Matrix::init_zero(
                self.weights_t.height,
                self.weights_t.width,
            ))
            .add_two_matrices_inplace(&input);

        match &self.second_moment_weight {
            Some(second_moment) => second_moment.div(1.0 - (beta2.powi(iteration))),
            None => panic!("Weight velocity should be initalized at this point"),
        }
    }

    fn compute_corrected_second_moment_biases(
        &mut self,
        mut input: Matrix,
        beta2: f64,
        iteration: i32,
    ) -> Matrix {
        // second_moment (t+1) = second_momentu(t) * beta2 + gradient(t+1)² * (1 - beta1)
        input.pow_inplace(2);
        input.mult_inplace(1.0 - beta2);
        self.second_moment_biase
            .get_or_insert(Matrix::init_zero(self.biases.height, self.biases.width))
            .mult_inplace(beta2);
        self.second_moment_biase
            .get_or_insert(Matrix::init_zero(self.biases.height, self.biases.width))
            .add_two_matrices_inplace(&input);

        match &self.second_moment_biase {
            Some(second_moment) => second_moment.div(1.0 - (beta2.powi(iteration))),
            None => panic!("Biase velocity should be initalized at this point"),
        }
    }
}

//unit test adam optimizer
#[cfg(test)]
mod tests {
    use crate::{optimizer::Optimizer, parse_test_csv::parse_test_csv};

    use super::Layer;

    #[test]
    fn test_adam_optimizer() {
        let test_data = parse_test_csv("tests/test_data/adam_test.csv".to_string());
        let mut layer = Layer::init_with_data(test_data[0].clone(), test_data[1].clone(), true);

        layer.first_moment_weight = Some(test_data[4].clone());
        layer.first_moment_biase = Some(test_data[5].clone());
        layer.second_moment_weight = Some(test_data[6].clone());
        layer.second_moment_biase = Some(test_data[7].clone());

        let adam = Optimizer::Adam {
            learning_step: 0.001,
            beta1: 0.9,
            beta2: 0.999,
        };
        let iteration = 7;

        layer.update_weigths(test_data[2].clone(), &adam, iteration);
        layer.update_biases(test_data[3].clone(), &adam, iteration);

        assert!(
            layer.weights_t.is_equal(&test_data[16], 10),
            "Adam test : the updated weights don't have the expected values"
        );
        assert!(
            layer.biases.is_equal(&test_data[17], 10),
            "Adam test : the updated biases don't have the expected values"
        );
    }
}
