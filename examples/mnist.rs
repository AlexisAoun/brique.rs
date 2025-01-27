use brique::layers::*;
use brique::matrix::*;
use brique::model::*;
use brique::optimizer::Optimizer;
use brique::save_load::*;
use brique::utils::*;

fn main() {
    training();
}

pub fn testing() {
    println!("extracting mnist data...");
    let labels: Matrix = extract_labels("examples/mnist_data/t10k-labels.idx1-ubyte");
    let mut images: Matrix = extract_images("examples/mnist_data/t10k-images.idx3-ubyte");
    println!("extraction done");

    images.normalize();
    println!("h {}", images.height);
    println!("w {}", images.width);

    println!("loading pre-trained model...");
    let mut model: Model = load_model("mnist_model_128x128_third_try".to_string()).unwrap();

    // println!("model layers : {}", model.layers.len());
    // println!("model layers1: {}", model.layers[0].weights_t.height);
    // println!("model layers1: {}", model.layers[0].weights_t.width);
    // println!("model layers2: {}", model.layers[1].weights_t.height);
    // println!("model layers2: {}", model.layers[1].weights_t.width);
    // println!("model layers3: {}", model.layers[2].weights_t.height);
    // println!("model layers3: {}", model.layers[2].weights_t.width);
    //
    // println!("model optimizer: {:?}", model.optimizer);
    println!("evaluating...");
    let score = model.evaluate(&images, false);
    let acc = model.accuracy(&score, &labels);

    println!("acc : {}", acc);
}

pub fn training() {
    println!("extracting mnist data...");
    let labels: Matrix = extract_labels("examples/mnist_data/train-labels.idx1-ubyte");
    let mut images: Matrix = extract_images("examples/mnist_data/train-images.idx3-ubyte");
    println!("extraction done");

    images.normalize();
    println!("h {}", images.height);
    println!("w {}", images.width);

    //_print_a_number(labels, images, 159);
    //

    let layer1 = Layer::init(28 * 28, 128, true);
    let layer2 = Layer::init(128, 128, true);
    let layer3 = Layer::init(128, 128, true);
    let layer4 = Layer::init(128, 10, false);

    let optimizer = Optimizer::Adam {
        learning_step: 0.001,
        beta1: 0.9,
        beta2: 0.999,
    };

    let mut model = Model::init(vec![layer1, layer2, layer3, layer4], optimizer, 0.005);

    model.train(&images, &labels, 128, 50, 2000, 100, false, false);
}

fn _print_a_number(labels: Matrix, images: Matrix, v: usize) {
    println!("{}", labels.get(0, v));

    for i in 0..28 * 28 {
        if images.get(v, i) > 0.5 && images.get(v, i) < 1.0 {
            if i % 28 == 0 {
                print!("\n");
            }

            if images.get(v, i) > 0.5 && images.get(v, i) < 0.75 {
                print!("-");
            } else {
                print!("*");
            }
        } else {
            if i % 28 == 0 {
                print!("\n");
            }
            print!("_");
        }
    }
}
