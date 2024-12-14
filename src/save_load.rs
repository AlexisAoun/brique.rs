use crate::matrix::Matrix;
use core::panic;
use std::{collections::HashMap, fs};

// CAT
const START_OF_OBJECT_MAGIC_NUMBER: [u8; 3] = [67, 65, 84];
// COOKIE
//const START_OF_FILE_MAGIC_NUMBER : [u8; 6] = [67, 79, 79, 75, 73, 69];

struct LookupStructBinaryId {
    lookup_table: HashMap<String, u8>,
}

impl LookupStructBinaryId {
    pub fn init() -> LookupStructBinaryId {
        let mut lookup_table = LookupStructBinaryId {
            lookup_table: HashMap::new(),
        };

        lookup_table.lookup_table.insert("Matrix".to_string(), 0);
        lookup_table.lookup_table.insert("Layer".to_string(), 1);
        lookup_table.lookup_table.insert("Model".to_string(), 2);

        lookup_table
    }

    pub fn lookup(self, struct_name: &str) -> u8 {
        let value: Option<&u8> = self.lookup_table.get(struct_name);

        if value.is_none() {
            panic!("Key in struct id lookup table not found");
        }

        *value.unwrap()
    }
}

pub fn read_write() {
    let data: Vec<f64> = vec![0.031, 0.0, -245.6457, 69.69];
    let matrix_input = Matrix::init(2, 2, data);
    let mut matrix_input2 = Matrix::init_rand(100, 50);
    matrix_input2.transpose_inplace();

    let mut byte_steam: Vec<u8> = vec![];
    byte_steam.append(&mut matrix_to_binary(&matrix_input));
    byte_steam.append(&mut matrix_to_binary(&matrix_input2));

    let res_write = fs::write("test.brq", byte_steam);

    if res_write.is_ok() {
        println!("File written successfuly");
    } else {
        panic!("Failed to write file");
    }

    let output_of_reading_file: Vec<u8> = fs::read("test.brq").unwrap();
    let (matrix_output, offset) = binary_to_matrix(&output_of_reading_file, 0);
    let (matrix_output2, _) = binary_to_matrix(&output_of_reading_file, offset);

    matrix_output2.display();
    println!("res : {}", matrix_input.is_equal(&matrix_output, 10));
    println!("res : {}", matrix_input2.is_equal(&matrix_output2, 10));
}

pub fn f64_array_to_binary(input: &Vec<f64>) -> Vec<u8> {
    let mut binary: Vec<u8> = vec![];

    input
        .iter()
        .for_each(|v| binary.append(&mut v.to_be_bytes().to_vec()));

    binary
}

pub fn binary_to_f64_array(input: Vec<u8>) -> Vec<f64> {
    let sized_slice_chunks = input.chunks(8);

    let mut output: Vec<f64> = vec![];
    sized_slice_chunks.for_each(|v| {
        let float = v.try_into().expect("Error while parsing binary chunks");
        output.push(f64::from_be_bytes(float));
    });

    output
}

// id
// height usize (u64)
// width usize (u64)
// transposed bool
// data Vec<f64>
pub fn matrix_to_binary(input: &Matrix) -> Vec<u8> {
    let mut output: Vec<u8> = vec![];
    // forcing usize to 64bit, just in case we are running on a 32bit system
    // not sure its the best way to deal with this
    let mut height_binary = (input.height as u64).to_be_bytes().to_vec();
    let mut width_binary = (input.width as u64).to_be_bytes().to_vec();

    let id_lookup_table = LookupStructBinaryId::init();

    output.append(&mut START_OF_OBJECT_MAGIC_NUMBER.to_vec());
    output.push(id_lookup_table.lookup("Matrix"));
    output.push(input.transposed as u8);
    output.append(&mut height_binary);
    output.append(&mut width_binary);
    output.append(&mut f64_array_to_binary(&input.data));

    output
}

pub fn binary_to_matrix(byte_stream: &Vec<u8>, input_offset: usize) -> (Matrix, usize) {
    let mut offset = input_offset;

    assert!(
        offset < byte_stream.len(),
        "Save binary reading : Unexpected EOF"
    );
    assert_eq!(
        byte_stream[offset..offset + 3],
        START_OF_OBJECT_MAGIC_NUMBER,
        "Save binary reading : Binary start of object code not found, file may be corrupted"
    );
    offset += 3;
    let id_lookup_table = LookupStructBinaryId::init();

    assert!(
        offset < byte_stream.len(),
        "Save binary reading : Unexpected EOF"
    );
    assert_eq!(byte_stream[offset], id_lookup_table.lookup("Matrix"), "Save binary reading : Binary id code does not match the lookup table for the Matrix entry, file may be corrupted");
    offset += 1;

    assert!(
        offset < byte_stream.len(),
        "Save binary reading : Unexpected EOF"
    );
    let transposed: bool = byte_stream[offset] != 0;
    offset += 1;

    assert!(
        offset + 8 < byte_stream.len(),
        "Save binary reading : Unexpected EOF"
    );
    let height: usize =
        u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    assert!(
        offset + 8 < byte_stream.len(),
        "Save binary reading : Unexpected EOF"
    );
    let width: usize =
        u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    println!("offset : {}", offset);
    println!("height : {}, width : {}", height, width);
    let data_size: usize = height * width * 8;
    assert!(
        offset + data_size <= byte_stream.len(),
        "Save binary reading : Unexpected EOF"
    );
    if offset + data_size + 2 <= byte_stream.len() {
        assert_eq!(
            byte_stream[offset + data_size..offset + data_size + 3],
            START_OF_OBJECT_MAGIC_NUMBER,
            "Save binary reading : Binary start of object code not found, file may be corrupted"
        );
    } else {
        assert_eq!(
            offset + data_size,
            byte_stream.len(),
            "Save binary reading : Unexpected EOF"
        );
    }

    let data: Vec<f64> = binary_to_f64_array(byte_stream[offset..offset + data_size].to_vec());
    offset += data_size;

    let output_matrix = Matrix {
        height,
        width,
        transposed,
        data,
    };

    (output_matrix, offset)
}
