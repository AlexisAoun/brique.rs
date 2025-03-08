# brique.rs
## What is brique.rs ? 

Brique.rs is a multi-perceptron layer (MLP) Rust library developped from scratch with the use of no other libraries. 

Everything was made from scratch : 

- The matrix data structure
- All the linear algebra logic
- All the MLP logic 
- A binary encoder/decoder for saving/loading traind models
- A CSV parser for unit/integration testing

The only dependencies of the project are the rand and rand-dist libs. There is no reliable way to generate random number in the standard library of Rust, and having a solid random number generator (on a normal distribution with a pre-determined standard deviation) is crucial for the weights initializations.

## Features

- Build and train a MLP model 
- Activation functions : ReLu, Softmax
- Optimizers : SGD, Adam
- Easy-to-use API based on a builder pattern
- Save and load models with .brq file format

## But why ? 

For two main reasons : I love building things, which is why I chose coding in the first place, and most importantly, to learn.

I significantly improved my Rust skills. I have know a better comprehension of the borrowing system and the memory structure in general.
Before this project, I had a good theoretical grasp of how MLPs are structured and learn. Now, by writing and testing everything myself, I have a very deep understanding of every aspect of it.
Beyond that I tackled many different problems. I wrote a CSV parser to facilitate unit and end-to-end testing and created my own binary encoder/decoder with a custom file format to save trained models.

This experience as a whole made me a better engineer.

## Installation

Create a Rust project 

```sh
cargo init
```

Add the library to your dependency list, in the Cargo.toml file

```sh
[dependencies]
brique.rs = "0.2.0"
```

## Usage

### Basic use

Here's a simple model trained on a spiral dataset, consisting of 3 spirals

Taken from examples/spiral.rs

```rust
use brique::layers::*;
use brique::model_builder::ModelBuilder;
use brique::optimizer::Optimizer;
use brique::spiral::generate_spiral_dataset;

pub fn main() {
    // generating the spiral dataset points
    // 3000 points, spread into three classes (here a class = one spiral)
    let (data, labels) = generate_spiral_dataset(3000, 3);

    // Layer::init(number_of_inputs: u32, number_of_neurons_for_the_layer: u32, reLu: bool)
    // if the last arg is true, applies ReLu as the activation function
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
```

Run the program 

```sh
cargo run --release
```

⚠️ **Do not forget the --release flag. If you don't use it the program could be significantly slower**

### The MNIST example 

Here's how to train the MNIST dataset and save the trained model using a checkpoint on the best validation accuracy 

You can use the pre-written functions to extract the dataset

```rust 
use brique::checkpoint::Checkpoint;
use brique::layers::*;
use brique::matrix::*;
use brique::model::*;
use brique::model_builder::ModelBuilder;
use brique::optimizer::Optimizer;
use brique::save_load::*;
use brique::utils::*;

pub fn main() {
    println!("extracting mnist data...");
    let labels: Matrix = extract_labels("examples/mnist_data/train-labels.idx1-ubyte");
    let mut images: Matrix = extract_images("examples/mnist_data/train-images.idx3-ubyte");
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
```

Code example of how to load the model and test it 

```rust 
pub fn main() {
    println!("extracting mnist data...");
    let labels: Matrix = extract_labels("examples/mnist_data/t10k-labels.idx1-ubyte");
    let mut images: Matrix = extract_images("examples/mnist_data/t10k-images.idx3-ubyte");
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
```

You can also directly launch the above examples with 

```sh 
cargo run --example spiral
```
And 
```sh 
cargo run --example mnist
```
## The .brq binary file

| **Field**          | **Size (bytes)** | **Description**                     |
|--------------------|------------------|-------------------------------------|
| **Header**         |                  |                                     |
| Magic Number       | 6                | Fixed identifier "COOKIE"           |
| Version            | 1                | Library version                     |
| Length             | 8                | Total file size                     |
| **Model Data**     |                  |                                     |
| Start of Object    | 3                | Fixed identifier "CAT"              |
| Model ID           | 1                | Identifier for Model                |
| Learning Step      | 8                | f64 value                           |
| Number of Layers   | 8                | u64 value                           |
| Layers             | Variable         | Depends on the number of layers     |
| **Layer Data**     |                  | (Repeated for each layer)           |
| Start of Object    | 3                | Fixed identifier "CAT"              |
| Layer ID           | 1                | Identifier for Layer                |
| Activation (ReLU)  | 1                | bool as u8                          |
| Weights Matrix     | Variable         | Depends on matrix size              |
| Biases Matrix      | Variable         | Depends on matrix size              |
| **Matrix Data**    |                  | (Repeated for each matrix)          |
| Start of Object    | 3                | Fixed identifier "CAT"              |
| Matrix ID          | 1                | Identifier for Matrix               |
| Transposed         | 1                | bool as u8                          |
| Height             | 8                | u64 value                           |
| Width              | 8                | u64 value                           |
| Data               | Variable         | Depends on the number of elements   |

