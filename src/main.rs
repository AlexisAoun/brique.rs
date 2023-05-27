mod layers;
mod matrix;
mod utils;
mod tests;
mod loss;
mod activation;

use crate::activation::softmax;

use crate::layers::*;
use crate::matrix::*;
//use crate::utils::*;

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
    
    // let mut m4: Matrix = Matrix::new(3, 3);
    // let mut m5: Matrix = Matrix::new(3, 1);
    //
    // m4.data[0][0] = 3.0;
    // m4.data[0][1] = 7.0;
    // m4.data[0][2] = -3.2;
    //
    // m4.data[1][0] = 3.0;
    // m4.data[1][1] = 4.0;
    // m4.data[1][2] = 2.2;
    //
    // m4.data[2][0] = 0.0;
    // m4.data[2][1] = -0.4;
    // m4.data[2][2] = 2.6;

    // m5.data[0][0] = 1.0;
    // m5.data[1][0] = -1.0;
    // m5.data[2][0] = 1.0;
    //
    // m4.display();
    // let m4_bis = softmax(&m4);
    // m4_bis.display();
    //
    //
    // for i in 0..3 {
    //     let s: f64 = m4_bis.data[i].iter().sum();
    //     println!("{}",s);
    // }

    // let test2 = m4.add_value_to_all_columns(&m5);
    // test2.display();
    //
    // let test_layer: ActivationLayer = ActivationLayer {};
    //
    // let test = test_layer.forward(&m5);
    // test.display();

    // let test_compute_layer = ComputeLayer::init(2, 4);
    // let mut m6: Matrix = Matrix::new(1, 2);
    // m6.data[0][0] = 1.0;
    // m6.data[0][1] = 1.0;
    //
    // let output = test_compute_layer.forward(&m6);
    // test_compute_layer.weights_t.display();
    // test_compute_layer.biases.display();
    // m6.display();
    // output.display();

}
