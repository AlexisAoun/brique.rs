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
    end_to_end_model_test();
    

}

    fn end_to_end_model_test() {
        let number_of_layers = 3;
        let input_weights: Vec<Matrix> = parse_test_csv("test_input_weights.csv".to_string());
        let test_data: Vec<Matrix> = parse_test_csv("test_data.csv".to_string());
        let expected_params: Vec<Matrix> = parse_test_csv("expected_params.csv".to_string());

        assert_eq!(
            input_weights.len(),
            number_of_layers,
            "The input weight csv doesn't have the expected number of matrices {}",
            number_of_layers
        );
        assert_eq!(
            test_data.len(),
            2,
            "The test data csv doesn't have the expected number of matrices {}",
            number_of_layers
        );

        let layer1 = Layer::init_test(2, 3, true, input_weights[0].clone());
        let layer2 = Layer::init_test(3, 3, true, input_weights[1].clone());
        let layer3 = Layer::init_test(3, 3, false, input_weights[2].clone());

        let mut model = Model {
            layers: vec![layer1, layer2, layer3],
            lambda: 0.001,
            learning_step: 0.1,
            input: Matrix::new(2,2),
            input_label: Matrix::new(2,2),
            itermediate_evaluation_results: Vec::new(),
            softmax_output: Matrix::new(2,2),
            d_zs: Vec::new(),
            d_ws: Vec::new(),
            d_bs: Vec::new(),
            d_score: Matrix::new(2,3),
            loss: 0.0,
            reg_loss: 0.0,
            data_loss: 0.0
        };

        let network_history = model.train(&test_data[0], &test_data[1], 6, 2, true);

        //TODO decide how to manage the expected params csv
        // iter thru the network_history and do assert_eq using is_equal() from matrix

        let models: Vec<Model> = network_history.unwrap();

        let index: usize = 0;
        for model in models {
            
            println!("----------------------- new epoch --- w and b : ");
            model.layers.iter().for_each(|l| {l.weights_t.display(); l.biases.display();});
            println!("----------------------- input : ");
            model.input.display();
            println!("----------------------- input_label: ");
            model.input_label.display();
            println!("-----------------------evaluation : ");
            model.itermediate_evaluation_results.iter().for_each(|l| l.display());
            println!("-----------------------softmax_output : ");
            model.softmax_output.display();
            println!("-----------------------loss : ");
            println!("data_loss : {}, reg_loss : {}, loss : {}", model.data_loss, model.reg_loss, model.loss);
            println!("-----------------------d_score : ");
            model.d_score.display();
            println!("-----------------------d_zs : ");
            model.d_zs.iter().for_each(|l| l.display());
            println!("-----------------------d_ws : ");
            model.d_ws.iter().for_each(|l| l.display());
            println!("-----------------------d_bs : ");
            model.d_bs.iter().for_each(|l| l.display());
            // assert!(
            //     model.layers[0]
            //         .weights_t
            //         .is_equal(&expected_params[index / 3]),
            //     "Weights model {}, layer 1, not expected value",
            //     index
            // );
            // assert!(
            //     model.layers[0]
            //         .biases
            //         .is_equal(&expected_params[index / 3 + 1]),
            //     "Biases model {}, layer 1, not expected value",
            //     index
            // );
            //
            // assert!(
            //     model.layers[1]
            //         .weights_t
            //         .is_equal(&expected_params[index / 3 + 2]),
            //     "Weights model {}, layer 1, not expected value",
            //     index
            // );
            //
            // assert!(
            //     model.layers[1]
            //         .biases
            //         .is_equal(&expected_params[index / 3 + 3]),
            //     "Biases model {}, layer 1, not expected value",
            //     index
            // );
            //
            // assert!(
            //     model.layers[2]
            //         .weights_t
            //         .is_equal(&expected_params[index / 3 + 4]),
            //     "Weights model {}, layer 1, not expected value",
            //     index
            // );
            // assert!(
            //     model.layers[2]
            //         .biases
            //         .is_equal(&expected_params[index / 3 + 5]),
            //     "Biases model {}, layer 1, not expected value",
            //     index
            // );
        }
    }
fn test_csv_import() {
    let output : Vec<Matrix> = parse_test_csv("expected_params.csv".to_string());

    output.iter().for_each(|m| m.display());

}

fn spiral_dataset_test_debug() {
    println!("generating spiral dataset");

    let (data, labels) = generate_spiral_dataset(3, 3);

    let layer1 = Layer::init(2, 3, true);
    let layer2 = Layer::init(3, 3, true);
    let layer3 = Layer::init(3, 3, false);
    // let mut model = Model {
    //     layers: vec![layer1, layer2, layer3],
    //     lambda: 0.001,
    //     learning_step: 1.0,
    // };

    //model.train(&data, &labels, 3, 1, false);
}

fn spiral_dataset_test() {
    println!("generating spiral dataset");

    let (data, labels) = generate_spiral_dataset(100, 3);

    let layer1 = Layer::init(2, 100, true);
    let layer2 = Layer::init(100, 100, true);
    let layer3 = Layer::init(100, 3, false);
    // let mut model = Model {
    //     layers: vec![layer1, layer2, layer3],
    //     lambda: 0.001,
    //     learning_step: 1.0,
    // };

    //model.train(&data, &labels, 300, 10000, false);
}

fn testing() {
    println!("extracting...");
    let labels: Matrix = extract_labels("data/train-labels.idx1-ubyte");
    let images: Matrix = extract_images("data/train-images.idx3-ubyte");

    let normalized_images: Matrix = images.normalize();

    let layer1 = Layer::init(28 * 28, 64, true);
    let layer3 = Layer::init(64, 10, false);

    // let mut model = Model {
    //     layers: vec![layer1, layer3],
    //     lambda: 0.0001,
    //     learning_step: 0.001,
    // };

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
