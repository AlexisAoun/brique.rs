use crate::layers::*;
use crate::matrix::*;
use crate::activation::*;

pub struct Model {
    pub layers: Vec<Layer>,
    pub lambda: f64
}

impl Model {
    pub fn evaluate(&self, input: &Matrix) -> Matrix {
        let mut index = 0;
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
}
