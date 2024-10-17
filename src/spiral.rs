use crate::matrix::Matrix;
use rand::prelude::*;

pub fn generate_spiral_dataset(number_of_points: u32, number_of_classes: u32) -> (Matrix, Matrix) {
    let height: usize = (number_of_points * number_of_classes) as usize;
    let mut data: Matrix = Matrix::init_zero(height, 2);
    let mut labels: Matrix = Matrix::init_zero(1, height);

    for class in 0..number_of_classes {
        let r: Vec<f64> = linspace(0.1, 1.0, number_of_points);

        let a: Vec<f64> = linspace(
            class as f64 * 4.0,
            (class + 1) as f64 * 4.0,
            number_of_points,
        );
        let t: Vec<f64> = add_rand_to_vec(&a);

        populate_data(&mut data, &mut labels, &r, &t, class, number_of_points);
    }

    (data, labels)
}

pub fn linspace(start: f64, stop: f64, number: u32) -> Vec<f64> {
    assert!(
        stop > start,
        "stop value should be greater than start value"
    );
    let delta: f64 = stop - start;
    let increment: f64 = delta / number as f64;

    let mut output: Vec<f64> = Vec::new();
    for i in 0..number + 1 {
        let x: f64 = start + (increment * i as f64);
        output.push(x);
    }

    output
}

fn add_rand_to_vec(input_vec: &Vec<f64>) -> Vec<f64> {
    let mut output: Vec<f64> = Vec::new();
    for index in 0..input_vec.len() {
        let mut r: f64 = random();
        r *= 0.2;
        output.push(input_vec[index] + r);
    }

    output
}

fn populate_data(
    data: &mut Matrix,
    labels: &mut Matrix,
    r: &Vec<f64>,
    t: &Vec<f64>,
    class: u32,
    n: u32,
) {
    let n_u = n as usize;
    let class_u = class as usize;
    let mut index_2 = 0;
    for index in n_u * class_u..n_u + (n_u * class_u) {

        let x : f64 = t[index_2].sin() * r[index_2];
        let y : f64 = t[index_2].cos() * r[index_2];

        data.set(x, index, 0);
        data.set(y, index, 1);
        labels.set(class as f64, 0, index);

        index_2 += 1;
    }
}
