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

    pub fn update_params(&mut self, d_score: &Matrix, input: &Matrix) {
        let index: usize = self.layers.len();
        while index >= 0 {
            let layer: &mut Layer = &mut self.layers[index];
            let previous_layer: Option<&mut Layer>;

            if index > 0 {
                previous_layer = Some(&mut self.layers[index-1]);
            } 

            if index == self.layers.len() - 1 {
                let z_minus_1: Matrix = match previous_layer {
                    None => input.clone(),
                    Some(layer) => layer.output.clone()
                };

                let d_w: Matrix = z_minus_1.t().dot(d_score);
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

                println!("loss : {}", loss);


                println!("score : ");
                score.display();

                
                println!("labels : ");
                batch_label.display();

                println!("d_score : ");
                let d_score = Model::compute_d_score(&score, &batch_label);
                self.update_params(&d_score, &batch_data);
            }
        }
    }
}
