use crate::matrix::Matrix;
use std::fs;

pub fn f64_to_u8() {

    let input : Vec<f64> = vec![0.031, 0.0, -245.6457, 69.69];

    let mut binary : Vec<u8> = vec![];

    input.iter().for_each(|v| binary.append(&mut v.to_be_bytes().to_vec()));

    println!("{:?}", binary);

    let res_write = fs::write("test.brq", binary);

    if res_write.is_ok() {
        println!("File written successfuly");
    } else {
        panic!("Failed to write file");
    }

    let output_of_reading_file : Vec<u8> = fs::read("test.brq").unwrap(); 
    let sized_slice_chunks = output_of_reading_file.chunks(8); 
  
    let mut output : Vec<f64> = vec![];
    sized_slice_chunks.for_each(|v| {
        let float = v.try_into().expect("Error while parsing binary chunks");
        output.push(f64::from_be_bytes(float));
    } );

    // let please = output_of_reading_file.try_into().expect("8");
    // let converted_output : f64 = f64::from_be_bytes(please);

    println!("{:?}", output_of_reading_file);
    println!("{:?}", output);

}

// height usize
// width usize
// transposed bool
// data Vec<f64>
pub fn matrix_to_binary(input : Matrix) -> Vec<u8> {
    let mut output : Vec<u8> = vec![];
    let mut height_binary = input.height.to_be_bytes().to_vec();

    output.append(&mut height_binary);


    output

}
