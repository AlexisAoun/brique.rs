mod matrice;
mod utils;

use crate::matrice::Matrice;
use crate::utils::{extract_images, extract_labels};

fn main() {
    let labels: Vec<u8> = extract_labels("data/train-labels.idx1-ubyte");
    let images: Matrice = extract_images("data/train-images.idx3-ubyte");

    println!("{}", labels[850]);

    for i in 0..28*28 {
        if images.data[850][i] > 0f64 {
            if i%28 == 0 { print!("\n"); }
            print!("*");    
        } else {
            if i%28 == 0 { print!("\n"); }
            print!("_");
        }
    }

}
