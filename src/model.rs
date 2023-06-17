use crate::activation::*;
use crate::layers::*;
use crate::loss::*;
use crate::matrix::*;
use crate::utils::*;

pub struct Model {
    pub layers: Vec<Layer>,
    pub lambda: f64,
    pub learning_step: f64
}

// all the variables begining with d (like d_score) are the derivative 
// of the loss function compared to said variable, so d_score is d Loss/ d Score 
// doing so for ease of read
impl Model {
    pub fn evaluate(&mut self, input: &Matrix) -> Matrix {
        let mut index: u32 = 0;
        let mut tmp: Matrix = Matrix::new(0, 0);
        for layer in self.layers.iter_mut() {
            if index == 0 {
                tmp = layer.forward(input);
            } else {
                tmp = layer.forward(&tmp);
            }

            //println!("evaluating index : {}", index);
            //tmp.display();
            index += 1;
        }
        softmax(&tmp)
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
                    output.data[r][c] = score.data[r][c] - 1.0;
                } else {
                    output.data[r][c] = score.data[r][c];
                }
            }
        }

        output
    }

    pub fn compute_d_relu(input: &Matrix) -> Matrix {
        let mut output: Matrix = Matrix::new(input.height, input.width);
        for r in 0..input.height {
            for c in 0..input.width {
                if input.data[r][c] < 0.0 {
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
                previous_layer = Some(self.layers[index-1].clone());
            } 

            let z_minus_1: Matrix = match previous_layer {
                None => input.clone(),
                Some(layer) => layer.output
            };

            let d_w: Matrix = z_minus_1.t().dot(&d_z).div(input.height as f64);
            d_w.add_two_matrices(&self.layers[index].weights_t.mult(self.lambda));

            let d_b: Matrix = d_z.sum_rows().div(input.height as f64);

            self.layers[index].update_weigths(&d_w, self.learning_step);
            self.layers[index].update_biases(&d_b, self.learning_step);

            //println!("layer number {}", index+1);
            // println!("weights $$$$$$$$$$$$$$$$");
            // self.layers[index].weights_t.display();
            // println!("biases $$$$$$$$$$$$$$");
            // self.layers[index].biases.display();


            // println!("Debug update params");
            // println!("layer number {}", index+1);
            // println!("d_z");
            // d_z.display();
            // println!("d_w");
            // d_w.display();
            // println!("d_b");
            // d_b.display();

            // println!("d_z height {} width {}", d_z.height, d_z.width);
            // println!("d_w height {} width {}", d_w.height, d_w.width);
            // println!("d_b height {} width {}", d_b.height, d_b.width);

            //println!("god have merccyyyyy");

            d_z = d_z.dot(&self.layers[index].weights_t.t());
            
            if index == 0 {
                break;
            }

            if self.layers[index - 1].activation {
                d_z = Model::compute_d_relu(&d_z);
            }

            index-=1;
        }

    }

    // the steps :
    // before every epoch :
    //  - shuffle dataset (use the algo of rand crate)
    //  - generate batch from shuffled dataset
    //  TODO i should really implement Matrix<T>
    //  TODO refactor it looks like ass
    pub fn train(&mut self, data: &Matrix, labels: &Matrix, batch_size: u32, epochs: u32) {
        for epoch in 0..epochs {
            let acc = self.accuracy(data, labels);
            println!("acc : {}", acc);
            println!("Iteration : {}", epoch);
            let index_table = generate_vec_rand_unique(data.height as u32);
            let index_matrix: Matrix = generate_batch_index(index_table, batch_size);

            for batch_indexes in index_matrix.data {
                let mut batch_data: Matrix = Matrix::new(batch_size as usize, data.width);
                let mut batch_label: Matrix = Matrix::new(1, batch_size as usize);

                for i in 0..batch_size as usize {
                    let index: usize = batch_indexes[i] as usize;
                    batch_data.data[i] = data.data[index].clone();
                    batch_label.data[0][i] = labels.data[0][index];
                }

                let score: Matrix = self.evaluate(&batch_data);
                let loss: f64 = self.compute_loss(&score, &batch_label);

                //score.display();
                println!("loss : {}", loss);

                let d_score = Model::compute_d_score(&score, &batch_label);
                self.update_params(&d_score, &batch_data);
            }
        }
    }

    pub fn accuracy(&mut self, data: &Matrix, labels: &Matrix) -> f64 {
        let score = self.evaluate(data);
        let answer = Self::answer(&score);

        let mut sum = 0;
        for index in 0..answer.width {
            if answer.data[0][index] == labels.data[0][index] {
                sum+=1;
            }
        }

        sum as f64 / answer.width as f64
    }

    pub fn answer(score: &Matrix) -> Matrix {
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
