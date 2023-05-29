use crate::layers::*;
use crate::matrix::*;
use crate::activation::*;
use crate::utils::*;

pub struct Model {
    pub layers: Vec<Layer>,
    pub lambda: f64
}

impl Model {
    pub fn evaluate(&self, input: &Matrix) -> Matrix {
        let mut index: u32 = 0;
        let mut tmp: Matrix = Matrix::new(0,0);
        for layer in self.layers.iter() {
            if index == 0 {
                tmp = layer.forward(input);
            } else {
                tmp = layer.forward(&tmp);
            }
            index+=1;
        }
        softmax(&tmp)
    }

    // the steps : 
    // before every epoch : 
    //  - shuffle dataset (use the algo of rand crate)
    //  - generate batch from shuffled dataset
    pub fn train(&self, data: &Matrix, labels: &Matrix, batch_size: u32, epochs: u32) {
        for epoch in 0..epochs {
            let index_table = generate_vec_rand_unique(data.height as u32);
        }
        
    }

}
