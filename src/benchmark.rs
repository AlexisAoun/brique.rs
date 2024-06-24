use std::time::Instant;
use crate::layers::*;
use crate::model::*;
use crate::spiral::*;

pub fn spiral_dataset_test() {
    println!("generating spiral dataset");

    let (data, labels) = generate_spiral_dataset(1000, 3);

    let layer1 = Layer::init(2, 300, true);
    let layer2 = Layer::init(300, 300, true);
    let layer3 = Layer::init(300, 3, false);

    let layers = vec![layer1, layer2, layer3];

    let mut model = Model::init(layers, 0.001, 0.01);

    model.train(&data, &labels, 50, 2, 500, false);
}

pub fn benchmark() {

    let start = Instant::now();

    spiral_dataset_test();

    let duration = start.elapsed();

    println!("The benchmark took {:?}", duration);

}
