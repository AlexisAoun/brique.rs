use crate::layers::*;
use crate::model::*;
use crate::optimizer::Optimizer;
use crate::spiral::*;
use std::time::Instant;

pub fn benchmark() {
    let start = Instant::now();

    spiral_dataset_test();

    let duration = start.elapsed();

    println!("The benchmark took {:?}", duration);
}

pub fn spiral_dataset_test() {
    println!("generating spiral dataset");

    let (data, labels) = generate_spiral_dataset(3000, 3);

    let layer1 = Layer::init(2, 30, true);
    let layer2 = Layer::init(30, 30, true);
    let layer3 = Layer::init(30, 3, false);

    let layers = vec![layer1, layer2, layer3];

    let optimizer = Optimizer::Adam {
        learning_step: 0.001,
        beta1: 0.9,
        beta2: 0.9,
    };
    let sgd = Optimizer::SGD {
        learning_step: 0.01,
    };
    let mut model = Model::init(layers, optimizer, 0.1);

    model.train(&data, &labels, 50, 2, 500, false, false);
}

// use crate::utils::*;
// use crate::matrix::*;
// will do later after optimizing the code

#[allow(dead_code)]
fn minst_test() {
    // println!("extracting...");
    // let labels: Matrix = extract_labels("data/train-labels.idx1-ubyte");
    // let images: Matrix = extract_images("data/train-images.idx3-ubyte");
    //
    // let normalized_images: Matrix = images.normalize();
    //
    // let _layer1 = Layer::init(28 * 28, 64, true);
    // let _layer3 = Layer::init(64, 10, false);

    // let mut model = Model {
    //     layers: vec![layer1, layer3],
    //     lambda: 0.0001,
    //     learning_step: 0.001,
    // };

    // println!("training...");
    // model.train(&normalized_images, &labels, 128, 5);

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
