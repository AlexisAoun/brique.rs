use crate::activation::*;
use crate::config::DEBUG;
use crate::matrix::*;
use crate::utils::*;

// note : we have directly the transpose of weights (hence the _t) to optimize computation
#[derive(Clone)]
pub struct Layer {
    pub weights_t: Matrix,
    pub biases: Matrix,
    pub activation: bool,
    pub output: Matrix
}

impl Layer {
    pub fn init(input_size: u32, size: u32, activation: bool) -> Layer {
        Layer {
            weights_t: Matrix::init_rand(input_size.try_into().unwrap(), size.try_into().unwrap()),
            biases: Matrix::new(1, size.try_into().unwrap()),
            activation,
            output: Matrix::new(0, 0)
        }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut tmp_output = input.dot(&self.weights_t);
        tmp_output = tmp_output.add_value_to_all_rows(&self.biases);

        if self.activation {
            tmp_output = self.activation(&tmp_output);
        } 

        self.output = tmp_output.clone();

        tmp_output
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

    pub fn update_weigths(&mut self, input: &Matrix, learning_step: f64) {
        if DEBUG {
            println!("##### Begining Update weights #####");
            println!("Weigths before : ");
            self.weights_t.display();
        }

        self.weights_t = self.weights_t.add_two_matrices(&input.mult(learning_step * -1.0));

        if DEBUG {
            println!("Weigths after : ");
            self.weights_t.display();
            println!("##### Ending Update weights #####");
        }
    }

    pub fn update_biases(&mut self, input: &Matrix, learning_step: f64) {
        if DEBUG {
            println!("##### Begining Update biases #####");
            println!("biases before : ");
            self.biases.display();
        }

        self.biases = self.biases.add_two_matrices(&input.mult(learning_step * -1.0));

        if DEBUG {
            println!("biases after : ");
            self.biases.display();
            println!("##### Ending Update biases #####");
        }
    }
}
