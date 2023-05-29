use crate::activation::*;
use crate::layers::*;
use crate::loss::*;
use crate::matrix::*;
use crate::utils::*;

pub struct Model {
    pub layers: Vec<Layer>,
    pub lambda: f64,
}

impl Model {
    pub fn evaluate(&self, input: &Matrix) -> Matrix {
        let mut index: u32 = 0;
        let mut tmp: Matrix = Matrix::new(0, 0);
        for layer in self.layers.iter() {
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

    // the steps :
    // before every epoch :
    //  - shuffle dataset (use the algo of rand crate)
    //  - generate batch from shuffled dataset
    //  TODO i should really implement Matrix<T>
    //  TODO refactor it looks like ass
    pub fn train(&self, data: &Matrix, labels: &Matrix, batch_size: u32, epochs: u32) {
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
                // TODO all the backprop thingy
            }
        }
    }
}
