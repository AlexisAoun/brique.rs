use crate::matrix::Matrix;
use rand::prelude::*;

fn generate_spiral_dataset(number_of_points: u32, number_of_classes: u32) -> (Matrix, Matrix) {
    let height: usize = (number_of_points*number_of_classes) as usize;
    let mut data: Matrix = Matrix::new(height, 2);
    let mut labels: Matrix = Matrix::new(1, height);

    for class in 0..number_of_classes {
        let r: Vec<f64> = linspace(0.0, 1.0, number_of_points);

        let rand: f64 = random();
        let a: Vec<f64> = linspace(class as f64 *4.0, (class + 1) as f64 * 4.0, number_of_points);
        let t: Vec<f64> = add_rand_to_vec(&a);

    } 

    (data, labels)
}

pub fn linspace(start: f64, stop: f64, number: u32) -> Vec<f64> {
    assert!(stop > start, "stop value should be greater than start value");
    let delta: f64 = stop - start;
    let increment: f64 = delta / number as f64; 
    
    let mut output: Vec<f64> = Vec::new();
    for i in 0..number+1 {
        let x: f64 = start + (increment * i as f64);
        output.push(x);
    }

    output
}

fn add_rand_to_vec(input_vec: &Vec<f64>) -> Vec<f64> {
    let mut output: Vec<f64> = Vec::new();
    for index in 0..input_vec.len() {
        let mut r: f64 = random();
        r*=0.2;
        output.push(input_vec[index] + r);
    }

    output
}

fn populate_data(data: &mut Matrix, r: &Vec<f64>, t: &Vec<f64> , class: u32, n: u32) {
    let index_delta: usize = data.height;
    for index in index_delta*class as usize..index_delta + index_delta*class as usize {
        data.data[index][0] = t[index].sin() * r[index];
        data.data[index][1] = t[index].cos() * r[index];
    }
}
