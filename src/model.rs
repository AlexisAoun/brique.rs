use crate::activation::*;
use crate::layers::*;
use crate::loss::*;
use crate::matrix::*;
use crate::optimizer::*;
use crate::utils::*;

#[derive(Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
    pub lambda: f64,
    pub optimizer: Optimizer,

    // these elements are stored in the struct for debugging purposes
    // only if debug arg is true
    pub layers_debug: Option<Vec<Layer>>,
    pub input: Option<Matrix>,
    pub input_label: Option<Matrix>,
    pub itermediate_evaluation_results: Option<Vec<Matrix>>,
    pub softmax_output: Option<Matrix>,
    pub data_loss: Option<f64>,
    pub reg_loss: Option<f64>,
    pub loss: Option<f64>,
    pub d_score: Option<Matrix>,
    pub d_zs: Option<Vec<Matrix>>,
    pub d_ws: Option<Vec<Matrix>>,
    pub d_bs: Option<Vec<Matrix>>,
}

// all the variables begining with d (like d_score) are the derivative
// of the loss function compared to said variable, so d_score is d Loss/ d Score
// doing so for ease of read
impl Model {
    pub fn init(layers: Vec<Layer>, optimizer: Optimizer, lambda: f64) -> Model {
        let output = Model {
            layers,
            lambda,
            optimizer,
            layers_debug: None,
            input: None,
            input_label: None,
            itermediate_evaluation_results: None,
            softmax_output: None,
            d_zs: None,
            d_ws: None,
            d_bs: None,
            d_score: None,
            loss: None,
            reg_loss: None,
            data_loss: None,
        };

        output
    }

    pub fn evaluate(&mut self, input: &Matrix, debug: bool) -> Matrix {
        for index in 0..self.layers.len() {
            if index == 0 {
                self.layers[0].forward(input, false);
            } else {
                let (l, r) = self.layers.split_at_mut(index);
                r[0].forward(&l[index - 1].output, false);
            }

            if debug {
                self.itermediate_evaluation_results
                    .get_or_insert(Vec::new())
                    .push(self.layers[index].output.clone());
            }
        }

        let output = softmax(&self.layers[self.layers.len() - 1].output);

        if debug {
            self.softmax_output = Some(output.clone());
        }

        output
    }

    // implementing cross-entropy and L2 regulariztion
    pub fn compute_loss(&mut self, output: &Matrix, labels: &Matrix, debug: bool) -> f64 {
        if debug {
            self.data_loss = Some(cross_entropy(output, labels));
            self.reg_loss = Some(l2_reg(&self.layers, self.lambda));
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

    pub fn update_params(&mut self, d_score: Matrix, input: Matrix, debug: bool) {
        let mut index: usize = self.layers.len() - 1;
        let mut d_z: Matrix = d_score;

        loop {
            if index > 0 {
                let (l, r) = self.layers.split_at_mut(index);
                d_z = r[0].backprop(
                    &d_z,
                    &l[index - 1].output,
                    l[index - 1].activation,
                    self.lambda,
                    &self.optimizer,
                    false,
                    debug,
                    &mut self.d_ws,
                    &mut self.d_bs,
                    &mut self.d_zs,
                );
            } else if index == 0 {
                self.layers[index].backprop(
                    &d_z,
                    &input,
                    false,
                    self.lambda,
                    &self.optimizer,
                    true,
                    debug,
                    &mut self.d_ws,
                    &mut self.d_bs,
                    &mut self.d_zs,
                );
                break;
            }

            index -= 1;
        }
    }

    // the steps :
    // before every epoch :
    //  - shuffle dataset (use the algo of rand crate)
    //  - generate batch from shuffled dataset
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
                    self.d_score = Some(d_score.clone());
                    self.input = Some(batch_data.clone());
                    self.input_label = Some(batch_label.clone());
                    self.loss = Some(loss);
                    self.layers_debug = Some(self.layers.clone());
                }

                self.update_params(d_score, batch_data, debug);

                if debug {
                    network_history.get_or_insert(Vec::new()).push(self.clone());
                    self.itermediate_evaluation_results = None;
                    self.d_zs = None;
                    self.d_bs = None;
                    self.d_ws = None;
                }

                batch_number += 1;

                if batch_number % 5 == 0 && !debug && !silent_mode {
                    let score_validation: Matrix = self.evaluate(&validation_data, false);
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
}
