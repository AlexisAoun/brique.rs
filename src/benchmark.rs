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

    model.train(&data, &labels, 50, 2, 500, 10, false, false);

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
