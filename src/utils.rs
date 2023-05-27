use crate::Matrix;
use std::fs::read;

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
    let mut output: Matrix = Matrix::new(1, slice.len());

    output.data[0] = slice.into_iter().map(|x| x as f64).collect();

    output
}

pub fn extract_images(path: &str) -> Matrix {
    let res: Vec<u8> = read(path).unwrap();

    check_image_file_header(&res);

    let array_size: u32 = convert_4_bytes_to_u32_big_endian(res[4..8].to_vec());
    let array_size_row: u32 = convert_4_bytes_to_u32_big_endian(res[8..12].to_vec());
    let array_size_column: u32 = convert_4_bytes_to_u32_big_endian(res[12..16].to_vec());

    let pixels_per_image: u32 = array_size_row * array_size_column;

    let mut output: Matrix = Matrix::new(
        pixels_per_image.try_into().unwrap(),
        array_size.try_into().unwrap(),
    );
    let mut index = 0;

    for i in res[16..].to_vec() {
        let x: usize = (index / pixels_per_image).try_into().unwrap();
        let y: usize = (index % pixels_per_image).try_into().unwrap();
        output.data[y][x] = i as f64;
        index += 1;
    }

    output
}
