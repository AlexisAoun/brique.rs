use brique::checkpoint::Checkpoint;
use brique::layers::*;
use brique::matrix::*;
use brique::model::*;
use brique::model_builder::ModelBuilder;
use brique::optimizer::Optimizer;
use brique::save_load::*;
use brique::utils::*;

fn main() {
    training();
}

pub fn testing() {
    println!("extracting mnist data...");
    let labels: Matrix = extract_labels("t10k-labels.idx1-ubyte");
    let mut images: Matrix = extract_images("t10k-images.idx3-ubyte");
    println!("extraction done");

    images.normalize();
    println!("number of images {}", images.height);
    println!("number of pixels in each image {}", images.width);

    println!("loading pre-trained model...");
    let mut model: Model = load_model("mnist_128x128".to_string()).unwrap();

    println!("evaluating...");
    let score = model.evaluate(&images, false);
    let acc = model.accuracy(&score, &labels);

    println!("acc : {}", acc);
}

pub fn training() {
    println!("extracting mnist data...");
    let labels: Matrix = extract_labels("train-labels.idx1-ubyte");
    let mut images: Matrix = extract_images("train-images.idx3-ubyte");
    println!("extraction done");

    images.normalize();
    println!("number of images {}", images.height);
    println!("number of pixels in each image {}", images.width);

    ModelBuilder::new()
        .add_layer(Layer::init(28 * 28, 128, true))
        .add_layer(Layer::init(128, 128, true))
        .add_layer(Layer::init(128, 10, false))
        .optimizer(Optimizer::Adam {
            learning_step: 0.001,
            beta1: 0.9,
            beta2: 0.999,
        })
        .l2_reg(0.001)
        .checkpoint(Checkpoint::ValAcc {
            save_path: "mnist_128x128".to_string(),
        })
        .verbose(10, false)
        .build_and_train(&images, &labels, 128, 10, 2000);
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
