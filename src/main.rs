mod activation;
mod layers;
mod loss;
mod matrix;
mod model;
mod tests;
mod utils;
mod spiral;
mod draw_spiral;
mod config;
mod log_into_csv;

use crate::layers::*;
use crate::matrix::*;
use crate::model::*;
use crate::utils::*;
use crate::spiral::*;
use crate::draw_spiral::*;
use crate::log_into_csv::*;


fn main() {
    spiral_dataset_test();
}

fn test_csv() {
    let mut m4: Matrix = Matrix::new(3, 3);

    m4.data[0][0] = 3.0;
    m4.data[0][1] = 7.0;
    m4.data[0][2] = -3.2;

    m4.data[1][0] = 3.0;
    m4.data[1][1] = 4.0;
    m4.data[1][2] = 2.2;

    m4.data[2][0] = 0.0;
    m4.data[2][1] = -0.4;
    m4.data[2][2] = 2.6;

    log_into_csv::log_matrix_into_csv("matrice 1", &m4);
    log_into_csv::log_matrix_into_csv("the same matrix", &m4);
}

fn spiral_dataset_test_debug() {
    println!("generating spiral dataset");

    let (data, labels) = generate_spiral_dataset(3, 3);

    let layer1 = Layer::init(2, 3, true);
    let layer2 = Layer::init(3, 3, false);
    let mut model = Model {
        layers: vec![layer1, layer2],
        lambda: 0.001,
        learning_step: 1.0
    };

    model.train(&data, &labels, 3, 1);
}

fn spiral_dataset_test() {
    println!("generating spiral dataset");

    let (data, labels) = generate_spiral_dataset(100, 3);

    let layer1 = Layer::init(2, 100, true);
    let layer2 = Layer::init(100, 3, false);
    let mut model = Model {
        layers: vec![layer1, layer2],
        lambda: 0.001,
        learning_step: 1.0
    };

    model.train(&data, &labels, 300, 10000);
}

fn testing() {
    println!("extracting...");
    let labels: Matrix = extract_labels("data/train-labels.idx1-ubyte");
    let images: Matrix = extract_images("data/train-images.idx3-ubyte");

    let normalized_images : Matrix = images.normalize();

    let layer1 = Layer::init(28 * 28, 64, true);
    let layer3 = Layer::init(64, 10, false);

    let mut model = Model {
        layers: vec![layer1, layer3],
        lambda: 0.0001,
        learning_step: 0.001,
    };

    println!("training...");
    model.train(&normalized_images, &labels, 128, 5);

    // for i in 0..20 {
    //     test_imags.data[i] = images.data[i].clone();
    //     test_labels.data[0][i] = labels.data[0][i];
    // }
    //
    // println!("{}", labels.data[0][469]);
    //
    // for i in 0..28 * 28 {
    //     if normalized_images.data[1111][i] > 0.5 && normalized_images.data[1111][i] < 1.0 {
    //         if i % 28 == 0 {
    //             print!("\n");
    //         }
    //
    //         if normalized_images.data[1111][i] > 0.5 && normalized_images.data[1111][i] < 0.75  {
    //             print!("-");
    //         } else {
    //             print!("*");
    //         }
    //     } else {
    //         if i % 28 == 0 {
    //             print!("\n");
    //         }
    //         print!("_");
    //     }
    // }
}
