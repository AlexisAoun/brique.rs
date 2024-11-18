use crate::activation::*;
use crate::layers::*;
use crate::loss::*;
use crate::matrix::*;
use crate::utils::*;

#[derive(Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
    pub lambda: f64,
    pub learning_step: f64,

    // these elements are stored in the struct for debugging purposes
    // only if debug arg is true
    pub layers_debug: Vec<Layer>,
    pub input: Matrix,
    pub input_label: Matrix,
    pub itermediate_evaluation_results: Vec<Matrix>,
    pub softmax_output: Matrix,
    pub data_loss: f64,
    pub reg_loss: f64,
    pub loss: f64,
    pub d_score: Matrix,
    pub d_zs: Vec<Matrix>,
    pub d_ws: Vec<Matrix>,
    pub d_bs: Vec<Matrix>,
}

// all the variables begining with d (like d_score) are the derivative
// of the loss function compared to said variable, so d_score is d Loss/ d Score
// doing so for ease of read
impl Model {
    pub fn init(layers: Vec<Layer>, lambda: f64, learning_step: f64) -> Model {
        let output = Model {
            layers,
            lambda,
            learning_step,
            layers_debug: Vec::new(),
            input: Matrix::init_zero(2, 2),
            input_label: Matrix::init_zero(2, 2),
            itermediate_evaluation_results: Vec::new(),
            softmax_output: Matrix::init_zero(2, 2),
            d_zs: Vec::new(),
            d_ws: Vec::new(),
            d_bs: Vec::new(),
            d_score: Matrix::init_zero(2, 3),
            loss: 0.0,
            reg_loss: 0.0,
            data_loss: 0.0,
        };

        output
    }
    pub fn evaluate(&mut self, input: &Matrix, debug: bool) -> Matrix {
        let mut index: u32 = 0;
        let mut tmp: Matrix = Matrix::init_zero(0, 0);
        for layer in self.layers.iter_mut() {
            if index == 0 {
                tmp = layer.forward(input, false);
            } else {
                tmp = layer.forward(&tmp, false);
            }

            if debug {
                self.itermediate_evaluation_results.push(tmp.clone());
            }

            index += 1;
        }

        let output = softmax(&tmp);

        if debug {
            self.softmax_output = output.clone();
        }

        output
    }

    // implementing cross-entropy and L2 regulariztion
    pub fn compute_loss(&mut self, output: &Matrix, labels: &Matrix, debug: bool) -> f64 {
        if debug {
            self.data_loss = cross_entropy(output, labels);
            self.reg_loss = l2_reg(&self.layers, self.lambda);
        }

        cross_entropy(output, labels) + l2_reg(&self.layers, self.lambda)
    }

    pub fn compute_d_score(score: &Matrix, labels: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::init_zero(score.height, score.width);
        for r in 0..score.height {
            for c in 0..score.width {
                //TODO make a choice, to divide or not to divide
                if labels.get(0, r) == c as f64 {
                    //output.data[r][c] = (score.data[r][c] - 1.0) / score.height as f64;
                    let v: f64 = score.get(r, c) - 1.0;

                    output.set(v, r, c);
                } else {
                    //output.data[r][c] = score.data[r][c] / score.height as f64;
                    let v: f64 = score.get(r, c);
                    output.set(v, r, c);
                }
            }
        }

        output
    }

    pub fn compute_d_relu(input: &Matrix, z_minus_1: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::init_zero(input.height, input.width);
        for r in 0..input.height {
            for c in 0..input.width {
                if z_minus_1.get(r, c) <= 0.0 {
                    output.set(0.0, r, c);
                } else {
                    output.set(input.get(r, c), r, c);
                }
            }
        }

        output
    }

    pub fn update_params(&mut self, d_score: &Matrix, input: &Matrix, debug: bool) {
        let mut index: usize = self.layers.len() - 1;
        //too much cloning going on here ?
        let mut d_z: Matrix = d_score.clone();

        loop {
            let mut previous_layer: Option<Layer> = None;

            if index > 0 {
                previous_layer = Some(self.layers[index - 1].clone());
            }

            let z_minus_1: Matrix = match previous_layer {
                None => input.clone(),
                Some(layer) => layer.output,
            };

            // compute W and B gradient
            let d_w_tmp: Matrix = z_minus_1.t().dot(&d_z);
            let d_w = d_w_tmp.add_two_matrices(&self.layers[index].weights_t.mult(self.lambda));

            let d_b: Matrix = d_z.sum_rows();

            if debug {
                self.d_zs.push(d_z.clone());
                self.d_ws.push(d_w.clone());
                self.d_bs.push(d_b.clone());
            }

            // compute gradient of the score for the next layer
            d_z = d_z.dot(&self.layers[index].weights_t.t());
            if index > 0  {
                if self.layers[index - 1].activation {
                    d_z = Model::compute_d_relu(&d_z, &z_minus_1);
                }
            }

            self.layers[index].update_weigths(&d_w, self.learning_step);
            self.layers[index].update_biases(&d_b, self.learning_step);

            if index == 0 {
                break;
            }

            index -= 1;
        }
    }

    // the steps :
    // before every epoch :
    //  - shuffle dataset (use the algo of rand crate)
    //  - generate batch from shuffled dataset
    //  TODO i should really implement Matrix<T>
    //  TODO refactor it looks like ass
    pub fn train(
        &mut self,
        data: &Matrix,
        labels: &Matrix,
        batch_size: u32,
        epochs: u32,
        validation_dataset_size: usize,
        debug: bool,
        silent_mode: bool, // if true will not print anything
    ) -> Option<Vec<Model>> {
        let mut network_history: Option<Vec<Model>> = None;

        if debug {
            network_history = Some(Vec::new());
        }

        let mut index_table: Vec<u32>;
        let index_validation: Vec<u32>;
        let mut validation_data: Matrix =
            Matrix::init_zero(validation_dataset_size as usize, data.width);
        let mut validation_label: Matrix = Matrix::init_zero(1, validation_dataset_size as usize);

        // first step is to randomize the input data
        // and to create the validation dataset
        // if debugging mode is on, no validation and no randomization
        if !debug {
            index_table = generate_vec_rand_unique(data.height as u32);

            index_validation = index_table[0..validation_dataset_size].to_vec();
            index_table.drain(0..validation_dataset_size);

            for i in 0..validation_dataset_size as usize {
                let index: usize = index_validation[i] as usize;
                // TODO write test for validation dataset creation
                validation_data.set_row(&data.get_row(index), i);
                validation_label.set(labels.get(0, index), 0, i);
            }
        } else {
            index_table = (0..data.height as u32).collect();
        }

        for epoch in 0..epochs {
            let index_matrix: Matrix = generate_batch_index(&index_table, batch_size);

            let mut batch_number = 0;

            for batch_row in 0..index_matrix.height {
                let batch_indexes: Vec<f64> = index_matrix.get_row(batch_row);
                let mut batch_data: Matrix = Matrix::init_zero(batch_size as usize, data.width);
                let mut batch_label: Matrix = Matrix::init_zero(1, batch_size as usize);

                for i in 0..batch_size as usize {
                    let index: usize = batch_indexes[i] as usize;
                    batch_data.set_row(&data.get_row(index), i);
                    batch_label.set(labels.get(0, index), 0, i);
                }

                let score: Matrix = self.evaluate(&batch_data, debug);
                let loss: f64 = self.compute_loss(&score, &batch_label, debug);

                let d_score = Model::compute_d_score(&score, &batch_label);

                if debug {
                    self.d_score = d_score.clone();
                    self.input = batch_data.clone();
                    self.input_label = batch_label.clone();
                    self.loss = loss;
                    self.layers_debug = self.layers.clone();
                }

                self.update_params(&d_score, &batch_data, debug);

                if debug {
                    let mut tmp = network_history.expect("network_history should be initialized");
                    tmp.push(self.clone());
                    network_history = Some(tmp);

                    //reinit bebug vars
                    self.itermediate_evaluation_results = Vec::new();
                    self.d_zs = Vec::new();
                    self.d_ws = Vec::new();
                    self.d_bs = Vec::new();
                }

                batch_number += 1;

                if batch_number % 5 == 0 && !debug && !silent_mode {
                    let score_validation: Matrix = self.predict(&validation_data);
                    let loss_validation: f64 =
                        self.compute_loss(&score_validation, &validation_label, debug);
                    let acc_validation: f64 = self.accuracy(&validation_data, &validation_label);

                    println!(
                        "Iteration : {}, Batch_number : {}, Loss : {}, Acc : {}",
                        epoch, batch_number, loss_validation, acc_validation
                    );
                }
            }
        }

        network_history
    }

    pub fn accuracy(&mut self, data: &Matrix, labels: &Matrix) -> f64 {
        let score = self.evaluate(data, false);
        let answer = Self::evaluation_output(&score);

        let mut sum = 0;
        for index in 0..answer.width {
            if answer.get(0, index) == labels.get(0, index) {
                sum += 1;
            }
        }

        sum as f64 / answer.width as f64
    }

    pub fn evaluation_output(score: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::init_zero(1, score.height);
        for r in 0..score.height {
            let mut index_max: u32 = 0;
            let mut last_max: f64 = 0.0;
            //iter ?
            for c in 0..score.width {
                if score.get(r, c) > last_max {
                    last_max = score.get(r, c);
                    index_max = c as u32;
                }
            }

            output.set(index_max as f64, 0, r);
        }

        output
    }

    pub fn predict(&mut self, input: &Matrix) -> Matrix {
        let mut index: u32 = 0;
        let mut tmp: Matrix = Matrix::init_zero(0, 0);
        for layer in self.layers.iter_mut() {
            if index == 0 {
                tmp = layer.forward(input, true);
            } else {
                tmp = layer.forward(&tmp, true);
            }

            index += 1;
        }

        let output = softmax(&tmp);

        output
    }
}
