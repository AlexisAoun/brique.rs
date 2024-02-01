use crate::activation::*;
use crate::config::DEBUG;
use crate::layers::*;
use crate::log_into_csv::log_matrix_into_csv;
use crate::loss::*;
use crate::matrix::*;
use crate::utils::*;

pub struct Model {
    pub layers: Vec<Layer>,
    pub lambda: f64,
    pub learning_step: f64,
}

// all the variables begining with d (like d_score) are the derivative
// of the loss function compared to said variable, so d_score is d Loss/ d Score
// doing so for ease of read
impl Model {
    pub fn evaluate(&mut self, input: &Matrix) -> Matrix {
        if DEBUG {
            log_matrix_into_csv("Begining evaluation : evaluation input", &input)
        }

        let mut index: u32 = 0;
        let mut tmp: Matrix = Matrix::new(0, 0);
        for layer in self.layers.iter_mut() {
            if index == 0 {
                tmp = layer.forward(input);
            } else {
                tmp = layer.forward(&tmp);
            }

            if DEBUG {
                let title = format!("evaluation index : {}, tmp evaluation matrix : ", index);
                log_matrix_into_csv(&title, &tmp);
            }

            index += 1;
        }

        let output = softmax(&tmp);

        if DEBUG {
            log_matrix_into_csv("Ending of evaluation, evaluation output : ", &output);
        }

        output
    }

    // implementing cross-entropy and L2 regulariztion
    pub fn compute_loss(&self, output: &Matrix, labels: &Matrix) -> f64 {
        cross_entropy(output, labels) + l2_reg(&self.layers, self.lambda)
    }

    pub fn compute_d_score(score: &Matrix, labels: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::new(score.height, score.width);
        for r in 0..score.height {
            for c in 0..score.width {
                if labels.data[0][r] == c as f64 {
                    output.data[r][c] = (score.data[r][c] - 1.0) / score.height as f64;
                } else {
                    output.data[r][c] = score.data[r][c] / score.height as f64;
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

    pub fn update_params(&mut self, d_score: &Matrix, input: &Matrix) {
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

            if DEBUG {
                let title = format!("layer index : {}, z_minus_1 : ", index);
                log_matrix_into_csv(&title, &z_minus_1);
                log_matrix_into_csv("d_z : ", &d_z);
            }

            let d_w_tmp: Matrix = z_minus_1.t().dot(&d_z);
            let d_w = d_w_tmp.add_two_matrices(&self.layers[index].weights_t.mult(self.lambda));

            let d_b: Matrix = d_z.sum_rows();

            if DEBUG {
                log_matrix_into_csv("d_w : ", &d_w);
                log_matrix_into_csv("d_b : ", &d_b);
                log_matrix_into_csv(
                    "reg * W = ",
                    &self.layers[index].weights_t.mult(self.lambda),
                );
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
    pub fn train(&mut self, data: &Matrix, labels: &Matrix, batch_size: u32, epochs: u32) {
        if DEBUG {
            log_matrix_into_csv("Begining training, data matrix : ", &data);
        }

        for epoch in 0..epochs {
            let mut avg_loss: f64 = 0.0;
            let mut avg_acc: f64 = 0.0;

            let mut sum_loss: f64 = 0.0;
            let mut sum_acc: f64 = 0.0;

            let index_table = generate_vec_rand_unique(data.height as u32);
            let index_matrix: Matrix = generate_batch_index(&index_table, batch_size);

            if DEBUG {
                let title = format!("Epoch number : {}, index matrix : ", epoch);
                log_matrix_into_csv(&title, &index_matrix);
            }

            let mut batch_number = 0;

            for batch_indexes in index_matrix.data {
                let mut batch_data: Matrix = Matrix::new(batch_size as usize, data.width);
                let mut batch_label: Matrix = Matrix::new(1, batch_size as usize);

                for i in 0..batch_size as usize {
                    let index: usize = batch_indexes[i] as usize;
                    batch_data.data[i] = data.data[index].clone();
                    batch_label.data[0][i] = labels.data[0][index];
                }

                if DEBUG {
                    let title = format!("Batch indexes : {:?}, batch data : ", batch_indexes);
                    log_matrix_into_csv(&title, &batch_data);
                    log_matrix_into_csv("batch_label : ", &batch_label);
                }

                let score: Matrix = self.evaluate(&batch_data);
                let loss: f64 = self.compute_loss(&score, &batch_label);
                let acc: f64 = self.accuracy(&batch_data, &batch_label);

                sum_loss += loss;
                sum_acc += acc;

                if batch_number == 0 {
                    avg_loss = loss;
                    avg_acc = acc;
                } else {
                    avg_loss = sum_loss / batch_number as f64;
                    avg_acc = sum_acc / batch_number as f64;
                }

                if batch_number % 100 == 0 {
                    println!(
                        "Iteration : {}, Batch_number : {}, Loss : {}, Acc : {}",
                        epoch, batch_number, avg_loss, avg_acc
                    );
                }

                let d_score = Model::compute_d_score(&score, &batch_label);

                if DEBUG {
                    log_matrix_into_csv("d_score : ", &d_score);
                }

                self.update_params(&d_score, &batch_data);
                batch_number += 1;
            }
        }
    }

    pub fn accuracy(&mut self, data: &Matrix, labels: &Matrix) -> f64 {
        let score = self.evaluate(data);
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
}
