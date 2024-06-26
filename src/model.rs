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
            input: Matrix::new(2, 2),
            input_label: Matrix::new(2, 2),
            itermediate_evaluation_results: Vec::new(),
            softmax_output: Matrix::new(2, 2),
            d_zs: Vec::new(),
            d_ws: Vec::new(),
            d_bs: Vec::new(),
            d_score: Matrix::new(2, 3),
            loss: 0.0,
            reg_loss: 0.0,
            data_loss: 0.0,
        };

        output
    }
    pub fn evaluate(&mut self, input: &Matrix, debug: bool) -> Matrix {
        let mut index: u32 = 0;
        let mut tmp: Matrix = Matrix::new(0, 0);
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
        let mut output: Matrix = Matrix::new(score.height, score.width);
        for r in 0..score.height {
            for c in 0..score.width {
                //TODO make a choice, to divide or not to divide
                if labels.data[0][r] == c as f64 {
                    //output.data[r][c] = (score.data[r][c] - 1.0) / score.height as f64;
                    output.data[r][c] = (score.data[r][c] - 1.0) as f64;
                } else {
                    //output.data[r][c] = score.data[r][c] / score.height as f64;
                    output.data[r][c] = score.data[r][c] as f64;
                }
            }
        }

        output
    }

    pub fn compute_d_relu(input: &Matrix, z_minus_1: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::new(input.height, input.width);
        for r in 0..input.height {
            for c in 0..input.width {
                if z_minus_1.data[r][c] <= 0.0 {
                    output.data[r][c] = 0.0;
                } else {
                    output.data[r][c] = input.data[r][c];
                }
            }
        }

        output
    }

    pub fn update_params(&mut self, d_score: &Matrix, input: &Matrix, debug: bool) {
        let mut index: usize = self.layers.len() - 1;
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

            let d_w_tmp: Matrix = z_minus_1.t().dot(&d_z);
            let d_w = d_w_tmp.add_two_matrices(&self.layers[index].weights_t.mult(self.lambda));

            let d_b: Matrix = d_z.sum_rows();

            if debug {
                self.d_zs.push(d_z.clone());
                self.d_ws.push(d_w.clone());
                self.d_bs.push(d_b.clone());
            }

            d_z = d_z.dot(&self.layers[index].weights_t.t());

            self.layers[index].update_weigths(&d_w, self.learning_step);
            self.layers[index].update_biases(&d_b, self.learning_step);

            if index == 0 {
                break;
            }

            if self.layers[index - 1].activation {
                d_z = Model::compute_d_relu(&d_z, &z_minus_1);
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
    ) -> Option<Vec<Model>> {
        let mut network_history: Option<Vec<Model>> = None;

        if debug {
            network_history = Some(Vec::new());
        }

        let mut index_table: Vec<u32>;
        let index_validation: Vec<u32>;
        let mut validation_data: Matrix = Matrix::new(validation_dataset_size as usize, data.width);
        let mut validation_label: Matrix = Matrix::new(1, validation_dataset_size as usize);

        if debug {
            index_table = (0..data.height as u32).collect();
        } else {
            index_table = generate_vec_rand_unique(data.height as u32);

            index_validation = index_table[0..validation_dataset_size].to_vec();
            index_table.drain(0..validation_dataset_size);

            for i in 0..validation_dataset_size as usize {
                let index: usize = index_validation[i] as usize;
                validation_data.data[i] = data.data[index].clone();
                validation_label.data[0][i] = labels.data[0][index];
            }
        }

        for epoch in 0..epochs {
            let index_matrix: Matrix = generate_batch_index(&index_table, batch_size);

            let mut batch_number = 0;

            for batch_indexes in index_matrix.data {
                let mut batch_data: Matrix = Matrix::new(batch_size as usize, data.width);
                let mut batch_label: Matrix = Matrix::new(1, batch_size as usize);

                for i in 0..batch_size as usize {
                    let index: usize = batch_indexes[i] as usize;
                    batch_data.data[i] = data.data[index].clone();
                    batch_label.data[0][i] = labels.data[0][index];
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

                if batch_number % 5 == 0 && !debug {
                    let score_validation: Matrix = self.predict(&validation_data);
                    let loss_validation: f64 = self.compute_loss(&score_validation, &validation_label, debug);
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
            if answer.data[0][index] == labels.data[0][index] {
                sum += 1;
            }
        }

        sum as f64 / answer.width as f64
    }

    pub fn evaluation_output(score: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::new(1, score.height);
        for r in 0..score.height {
            let mut index_max: u32 = 0;
            let mut last_max: f64 = 0.0;
            for c in 0..score.width {
                if score.data[r][c] > last_max {
                    last_max = score.data[r][c];
                    index_max = c as u32;
                }
            }

            output.data[0][r] = index_max as f64;
        }

        output
    }

    pub fn predict(&mut self, input: &Matrix) -> Matrix {
        let mut index: u32 = 0;
        let mut tmp: Matrix = Matrix::new(0, 0);
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
