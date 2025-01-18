use crate::layers::*;
use crate::model::*;
use crate::optimizer::Optimizer;
use crate::save_load;
use crate::save_load::*;
use crate::spiral::*;
use std::time::Instant;

pub fn benchmark() {
    let start = Instant::now();

    spiral_dataset_test();
    load_trained_spiral_model_and_test();

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
    let adam = Optimizer::Adam {
        learning_step: 0.05,
        beta1: 0.9,
        beta2: 0.999,
    };
    let sgd = Optimizer::SGD {
        learning_step: 0.001,
    };
    let mut model = Model::init(layers, adam, 0.001);

    model.train(&data, &labels, 50, 2, 500, false, false);

    save_load::save_model(&model, "spiral_model".to_string()).unwrap();
}

pub fn load_trained_spiral_model_and_test() {
    let (test_data, test_labels) = generate_spiral_dataset(3000, 3);
    let mut spiral_model = save_load::load_model("spiral_model".to_string()).unwrap();

    let acc_trained: f64 = spiral_model.accuracy(&test_data, &test_labels);

    let layer1 = Layer::init(2, 30, true);
    let layer2 = Layer::init(30, 30, true);
    let layer3 = Layer::init(30, 3, false);

    let layers = vec![layer1, layer2, layer3];
    let adam = Optimizer::Adam {
        learning_step: 0.05,
        beta1: 0.9,
        beta2: 0.999,
    };
    let sgd = Optimizer::SGD {
        learning_step: 0.001,
    };
    let mut model = Model::init(layers, adam, 0.001);
    let acc_not_trained: f64 = model.accuracy(&test_data, &test_labels);

    println!("acc trained : {}", acc_trained);
    println!("acc not trained : {}", acc_not_trained);
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
