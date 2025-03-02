use brique::checkpoint::Checkpoint;
use brique::layers::*;
use brique::matrix::*;
use brique::model::*;
use brique::model_builder::ModelBuilder;
use brique::optimizer::Optimizer;
use brique::save_load::*;
use brique::utils::*;

pub fn spiral_dataset_test() {
    // generating the spiral dataset points 
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
    
    let mut model = Model::init(layers, adam, 0.001);
    
    model.train(&data, &labels, 50, 2, 500, 10, false, false);
    
    save_load::save_model(&model, "spiral_model".to_string()).unwrap();
}
