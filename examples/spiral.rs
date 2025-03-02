use brique::layers::*;
use brique::model_builder::ModelBuilder;
use brique::optimizer::Optimizer;
use brique::spiral::generate_spiral_dataset;

pub fn main() {
    // generating the spiral dataset points
    // 3000 points, spread into three classes (here a class = one spiral)
    let (data, labels) = generate_spiral_dataset(3000, 3);

    // Layer::init(number_of_inputs: u32, number_of_neurons_for_the_layer: u32, reLu: bool)
    // if the last is arg, applies ReLu as the activation function
    // by default softmax is applied to the last layer

    // One point of the spiral dataset consists of a X and a Y
    // So the first layer has 2 inputs
    // The last layer has 3 neurons because we have 3 classes, and therefore we want 3 outputs

    // build and train
    // (data: &matrix, labels: &matrix, batch_size: u32, number_of_epochs: u32, size_of_the_validation_dataset, usize)
    let _ = ModelBuilder::new()
        .add_layer(Layer::init(2, 10, true))
        .add_layer(Layer::init(10, 10, true))
        .add_layer(Layer::init(10, 3, false))
        .optimizer(Optimizer::SGD {
            learning_step: 0.001,
        })
        .l2_reg(0.0001)
        .build_and_train(&data, &labels, 128, 10, 500);
}
