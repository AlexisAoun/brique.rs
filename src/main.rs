mod layers;
mod matrix;
mod utils;

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

    // let mut m4: Matrix = Matrix::new(3, 5);
    // let mut m5: Matrix = Matrix::new(3, 1);
    //
    // m4.data[0][0] = 3.0;
    // m4.data[1][0] = 7.0;
    // m4.data[2][0] = -3.2;
    //
    // m5.data[0][0] = 1.0;
    // m5.data[1][0] = -1.0;
    // m5.data[2][0] = 1.0;
    //
    // m4.display();
    // m5.display();
    //
    // let test2 = m4.add_value_to_all_columns(&m5);
    // test2.display();
    //
    // let test_layer: ActivationLayer = ActivationLayer {};
    //
    // let test = test_layer.forward(&m5);
    // test.display();

    let test_compute_layer = ComputeLayer::init(4, 10);
    test_compute_layer.weights.display();
    test_compute_layer.biases.display();
}
