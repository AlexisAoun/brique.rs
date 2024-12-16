use brique::{layers::Layer, model::Model, save_load::*};

fn main() {
    let layer1 = Layer::init(5, 10, true);
    let layer2 = Layer::init(10, 8, false);

    let model = Model::init(vec![layer1.clone(), layer2.clone()], 0.1, 0.0065);

    save_model(model, "model1".to_string()).unwrap();

    let model_retreived: Model = match load_model("model1".to_string()) {
        Ok(model) => model,
        Err(e) => panic!("Error : {}", e),
    };
    model_retreived.layers[0].weights_t.display();
    model_retreived.layers[1].weights_t.display();
    println!(
        "{}",
        model_retreived.layers[0]
            .weights_t
            .is_equal(&layer1.weights_t, 10)
    );
    println!(
        "{}",
        model_retreived.layers[1]
            .weights_t
            .is_equal(&layer2.weights_t, 10)
    );
    println!(
        "{}",
        model_retreived.layers[0]
            .biases
            .is_equal(&layer1.biases, 10)
    );
    println!(
        "{}",
        model_retreived.layers[1]
            .biases
            .is_equal(&layer2.biases, 10)
    );
}
