mod activation;
mod config;
mod draw_spiral;
mod layers;
mod log_into_csv;
mod loss;
mod matrix;
mod model;
mod parse_test_csv;
mod spiral;
mod tests;
mod utils;

use crate::layers::*;
use crate::log_into_csv::*;
use crate::matrix::*;
use crate::model::*;
use crate::parse_test_csv::*;
use crate::spiral::*;
use crate::utils::*;

fn main() {
    parse_test_csv();
}

fn spiral_dataset_test_debug() {
    println!("generating spiral dataset");

    let (data, labels) = generate_spiral_dataset(3, 3);

    let layer1 = Layer::init(2, 3, true);
    let layer2 = Layer::init(3, 3, true);
    let layer3 = Layer::init(3, 3, false);
    let mut model = Model {
        layers: vec![layer1, layer2, layer3],
        lambda: 0.001,
        learning_step: 1.0,
    };

    model.train(&data, &labels, 3, 1);
}

fn spiral_dataset_test() {
    println!("generating spiral dataset");

    let (data, labels) = generate_spiral_dataset(100, 3);

    let layer1 = Layer::init(2, 32, true);
    let layer2 = Layer::init(32, 32, true);
    let layer3 = Layer::init(32, 3, false);
    let mut model = Model {
        layers: vec![layer1, layer2, layer3],
        lambda: 0.001,
        learning_step: 1.0,
    };

    model.train(&data, &labels, 300, 10000);
}

fn testing() {
    println!("extracting...");
    let labels: Matrix = extract_labels("data/train-labels.idx1-ubyte");
    let images: Matrix = extract_images("data/train-images.idx3-ubyte");

    let normalized_images: Matrix = images.normalize();

    let layer1 = Layer::init(28 * 28, 64, true);
    let layer3 = Layer::init(64, 10, false);

    let mut model = Model {
        layers: vec![layer1, layer3],
        lambda: 0.0001,
        learning_step: 0.001,
    };

    println!("training...");
    //model.train(&normalized_images, &labels, 128, 5);

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
