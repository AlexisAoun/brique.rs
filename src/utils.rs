use crate::matrix::*;
use rand::seq::SliceRandom;
use std::fs::read;

pub fn generate_vec_rand_unique(size: u32) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let mut output: Vec<u32> = (0..size).collect();

    output.shuffle(&mut rng);
    output
}

// not the optimal way to return Matrix with f64s. can be optimised with matrix that accepts
// generic type
pub fn generate_batch_index(index_table: &Vec<u32>, batch_size: u32) -> Matrix {
    assert!(
        index_table.len() as u32 >= batch_size,
        "Batch size cannot be bigger than training dataset size"
    );
    assert!(batch_size > 0, "Batch size must be strictly positive");

    let mut number_of_batches: usize = index_table.len() / batch_size as usize;
    if index_table.len() % (batch_size as usize) != 0 {
        number_of_batches += 1;
    }
    let mut output: Matrix = Matrix::init_zero(number_of_batches, batch_size as usize);

    // i shouldnt code at 2am. bad things happen
    'outer: for i in 0..number_of_batches {
        for j in 0..batch_size as usize {
            let index: usize = (i * batch_size as usize) + j;
            if index < index_table.len() {
                output.set(index_table[index] as f64, i, j);
            } else {
                // just drop if uneven cant be bothered
                output.data.pop();
                output.height -= 1;
                break 'outer;
            }
        }
    }

    output
}

fn convert_4_bytes_to_u32_big_endian(bytes: Vec<u8>) -> u32 {
    assert_eq!(bytes.len(), 4, "byte array should be of size 4");
    let output: u32 = (bytes[0] as u32) * 2_u32.pow(24)
        + (bytes[1] as u32) * 2_u32.pow(16)
        + (bytes[2] as u32) * 2_u32.pow(8)
        + (bytes[3] as u32);

    output
}

fn check_label_file_header(array: &Vec<u8>) {
    // check out the documentation : http://yann.lecun.com/exdb/mnist/
    let expected_file_header: Vec<u8> = vec![0, 0, 8, 1];
    let array_size: u32 = convert_4_bytes_to_u32_big_endian(array[4..8].to_vec());

    assert_eq!(
        array[0..4].to_vec(),
        expected_file_header,
        "File incompatibility detected, are you sure you added the correct LABEL file ?"
    );
    assert_eq!(
        array_size,
        array.len() as u32 - 8,
        "File incompatibility detected, are you sure you added the correct LABEL file ?"
    );
}

fn check_image_file_header(array: &Vec<u8>) {
    // check out the documentation : http://yann.lecun.com/exdb/mnist/
    let expected_file_header: Vec<u8> = vec![0, 0, 8, 3];

    let array_size: u32 = convert_4_bytes_to_u32_big_endian(array[4..8].to_vec());
    let array_size_row: u32 = convert_4_bytes_to_u32_big_endian(array[8..12].to_vec());
    let array_size_column: u32 = convert_4_bytes_to_u32_big_endian(array[12..16].to_vec());

    assert_eq!(
        array_size * array_size_column * array_size_row,
        array.len() as u32 - 16,
        "File incompatibility detected, are you sure you added the correct IMAGE file ?"
    );
    assert_eq!(
        array[0..4].to_vec(),
        expected_file_header,
        "File incompatibility detected, are you sure you added the correct IMAGE file ?"
    );
}

pub fn extract_labels(path: &str) -> Matrix {
    let res: Vec<u8> = read(path).unwrap();
    check_label_file_header(&res);
    let slice: Vec<u8> = res[8..].to_vec();
    let mut output: Matrix = Matrix::init_zero(1, slice.len());

    slice.iter().enumerate().for_each(|(index, value)| {
        output.set(*value as f64, 0, index)
    });

    output
}

pub fn extract_images(path: &str) -> Matrix {
    let res: Vec<u8> = read(path).unwrap();

    check_image_file_header(&res);

    let array_size: u32 = convert_4_bytes_to_u32_big_endian(res[4..8].to_vec());
    let array_size_row: u32 = convert_4_bytes_to_u32_big_endian(res[8..12].to_vec());
    let array_size_column: u32 = convert_4_bytes_to_u32_big_endian(res[12..16].to_vec());

    let pixels_per_image: u32 = array_size_row * array_size_column;

    let mut output: Matrix = Matrix::init_zero(
        array_size.try_into().unwrap(),
        pixels_per_image.try_into().unwrap(),
    );
    let mut index = 0;

    for i in res[16..].to_vec() {
        let x: usize = (index / pixels_per_image).try_into().unwrap();
        let y: usize = (index % pixels_per_image).try_into().unwrap();
        output.set(i as f64, x, y);
        index += 1;
    }

    output
}
