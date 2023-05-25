mod matrix;
mod utils;

use crate::matrix::Matrix;
use crate::utils::{extract_images, extract_labels};

pub struct compute_layer {
    pub weights: Matrix,
    pub biases: Matrix,
}

impl compute_layer {
    pub fn forward(self, input: Matrix) -> Matrix {
        let mut output: Matrix = input.dot(self.weights);
        output = output.add_value_to_all_columns(self.biases);

        output
    }
}

fn main() {
    // let labels: Matrix = extract_labels("data/train-labels.idx1-ubyte");
    // let images: Matrix = extract_images("data/train-images.idx3-ubyte");
    //
    // println!("{}", labels.data[0][850]);
    //
    // for i in 0..28 * 28 {
    //     if images.data[i][850] > 0f64 {
    //         if i % 28 == 0 {
    //             print!("\n");
    //         }
    //         print!("*");
    //     } else {
    //         if i % 28 == 0 {
    //             print!("\n");
    //         }
    //         print!("_");
    //     }
    // }
    //
    // print!("\n");
    // let mut m1: Matrix = Matrix::new(2,2);
    // m1.data[0][0] = -3242.213;
    // m1.data[1][0] = 1242356.4245;
    // m1.data[0][1] = 41466.9088;
    // m1.data[1][1] = 0.0;
    // let mut m2: Matrix = Matrix::new(2,3);
    // m2.data[0][0] = 898.0;
    // m2.data[1][0] = -222.3;
    // m2.data[0][1] = -2467.9098;
    // m2.data[1][1] = 1356770.0;
    // m2.data[0][2] = -696969.69696;
    // m2.data[1][2] = 10.0;
    // m1.display();
    // m2.display();
    //
    // let m3 = m1.dot(m2);
    // m3.display();

    let mut m4: Matrix = Matrix::new(3,5);
    let mut m5: Matrix = Matrix::new(3,1);


    m4.data[0][0] = 3.0;
    m4.data[1][0] = 7.0;
    m4.data[2][0] = -3.2;

    m5.data[0][0] = 1.0;
    m5.data[1][0] = 1.0;
    m5.data[2][0] = 1.0;

    m4.display();
    m5.display();

    let m6 = m4.add_value_to_all_columns(m5);
    m6.display();
}
