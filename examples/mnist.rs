use brique::layers::*;
use brique::matrix::*;
use brique::model::*;
use brique::optimizer::Optimizer;
use brique::save_load::*;
use brique::utils::*;

fn main() {
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
        learning_step: 0.015,
        beta1: 0.9,
        beta2: 0.999,
    };

    let mut model = Model::init(vec![layer1, layer2, layer3, layer4], optimizer, 0.001);

    model.train(&images, &labels, 128, 5, 1000, 10, false, false);
    let _ = save_model(&model, "mnist_model4_128x128".to_string());
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
