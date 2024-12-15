use crate::{layers::Layer, matrix::Matrix, model::Model};
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
    let layer_1 = Layer {
        weights_t: matrix_input,
        biases: Matrix::init_rand(1, 5),
        activation: false,
        output: Matrix::init_zero(0, 0),
    };

    let layer_2 = Layer::init(7, 10, true);
    let model = Model::init(vec![layer_1.clone(), layer_2.clone()], 0.1, 2.0);

    let mut byte_steam: Vec<u8> = vec![];
    byte_steam.append(&mut model_to_binary(&model));

    let res_write = fs::write("test.brq", byte_steam);

    if res_write.is_ok() {
        println!("File written successfuly");
    } else {
        panic!("Failed to write file");
    }

    let output_of_reading_file: Vec<u8> = fs::read("test.brq").unwrap();
    let model_output = binary_to_model(&output_of_reading_file, 0);

    assert_eq!(layer_1.activation, model_output.layers[0].activation);
    layer_1.weights_t.display();
    model_output.layers[0].weights_t.display();

    layer_2.weights_t.display();
    model_output.layers[1].weights_t.display();
    assert_eq!(layer_2.activation, model_output.layers[1].activation);
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
        "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
    );
    assert_eq!(
        byte_stream[offset..offset + 3],
        START_OF_OBJECT_MAGIC_NUMBER,
        "Save binary reading - while attempting to decode a matrix : Binary start of object code not found, file may be corrupted"
    );
    offset += 3;
    let id_lookup_table = LookupStructBinaryId::init();

    assert!(
        offset < byte_stream.len(),
        "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
    );
    assert_eq!(byte_stream[offset], id_lookup_table.lookup("Matrix"), "Save binary reading - while attempting to decode a matrix : Binary id code does not match the lookup table for the Matrix entry, file may be corrupted");
    offset += 1;

    assert!(
        offset < byte_stream.len(),
        "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
    );
    let transposed: bool = byte_stream[offset] != 0;
    offset += 1;

    assert!(
        offset + 8 < byte_stream.len(),
        "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
    );
    let height: usize =
        u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    assert!(
        offset + 8 < byte_stream.len(),
        "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
    );
    let width: usize =
        u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    let data_size: usize = height * width * 8;
    assert!(
        offset + data_size <= byte_stream.len(),
        "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
    );
    if offset + data_size + 2 <= byte_stream.len() {
        assert_eq!(
            byte_stream[offset + data_size..offset + data_size + 3],
            START_OF_OBJECT_MAGIC_NUMBER,
            "Save binary reading - while attempting to decode a matrix : Binary start of object code not found, file may be corrupted"
        );
    } else {
        assert_eq!(
            offset + data_size,
            byte_stream.len(),
            "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
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

// weights : matrix
// biases : matrix
// activation : bool
pub fn layer_to_binary(input_layer: &Layer) -> Vec<u8> {
    let mut output: Vec<u8> = vec![];

    let id_lookup_table = LookupStructBinaryId::init();

    output.append(&mut START_OF_OBJECT_MAGIC_NUMBER.to_vec());
    output.push(id_lookup_table.lookup("Layer"));
    output.push(input_layer.activation as u8);
    output.append(&mut matrix_to_binary(&input_layer.weights_t));
    output.append(&mut matrix_to_binary(&input_layer.biases));

    output
}

pub fn binary_to_layer(byte_stream: &Vec<u8>, input_offset: usize) -> (Layer, usize) {
    let mut offset = input_offset;

    assert!(
        offset < byte_stream.len(),
        "Save binary reading - while attempting to decode a layer : Unexpected EOF"
    );
    assert_eq!(
        byte_stream[offset..offset + 3],
        START_OF_OBJECT_MAGIC_NUMBER,
        "Save binary reading - while attempting to decode a layer : Binary start of object code not found, file may be corrupted"
    );
    offset += 3;
    let id_lookup_table = LookupStructBinaryId::init();

    assert!(
        offset < byte_stream.len(),
        "Save binary reading - while attempting to decode a layer : Unexpected EOF"
    );
    assert_eq!(byte_stream[offset], id_lookup_table.lookup("Layer"), "Save binary reading - while attempting to decode a layer : Binary id code does not match the lookup table for the Layer entry, file may be corrupted");
    offset += 1;

    assert!(
        offset < byte_stream.len(),
        "Save binary reading - while attempting to decode a layer : Unexpected EOF"
    );
    let activation: bool = byte_stream[offset] != 0;
    offset += 1;

    let (weights_t, offset) = binary_to_matrix(byte_stream, offset);
    let (biases, offset) = binary_to_matrix(byte_stream, offset);

    let output_layer = Layer {
        weights_t,
        biases,
        activation,
        output: Matrix::init_zero(0, 0),
    };

    (output_layer, offset)
}

// learning step f64
// lambda f64
// number of layers
// layres Vec<Layer>
pub fn model_to_binary(input_model: &Model) -> Vec<u8> {
    let mut output: Vec<u8> = vec![];

    let id_lookup_table = LookupStructBinaryId::init();

    output.append(&mut START_OF_OBJECT_MAGIC_NUMBER.to_vec());
    output.push(id_lookup_table.lookup("Model"));
    output.append(&mut input_model.learning_step.to_be_bytes().to_vec());
    output.append(&mut input_model.lambda.to_be_bytes().to_vec());
    output.append(&mut (input_model.layers.len() as u64).to_be_bytes().to_vec());

    input_model
        .layers
        .iter()
        .for_each(|layer| output.append(&mut layer_to_binary(&layer)));

    output
}

pub fn binary_to_model(byte_stream: &Vec<u8>, input_offset: usize) -> Model {
    let mut offset = input_offset;

    assert!(
        offset < byte_stream.len(),
        "Save binary reading - while attempting to decode a model : Unexpected EOF"
    );
    assert_eq!(
        byte_stream[offset..offset + 3],
        START_OF_OBJECT_MAGIC_NUMBER,
        "Save binary reading - while attempting to decode a model : Binary start of object code not found, file may be corrupted"
    );
    offset += 3;
    let id_lookup_table = LookupStructBinaryId::init();

    assert!(
        offset < byte_stream.len(),
        "Save binary reading - while attempting to decode a model : Unexpected EOF"
    );
    assert_eq!(byte_stream[offset], id_lookup_table.lookup("Model"), "Save binary reading - while attempting to decode a model : Binary id code does not match the lookup table for the Layer entry, file may be corrupted");
    offset += 1;

    assert!(
        offset + 8 < byte_stream.len(),
        "Save binary reading - while attempting to decode a model : Unexpected EOF"
    );
    let learning_step: f64 =
        f64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap());
    offset += 8;

    assert!(
        offset + 8 < byte_stream.len(),
        "Save binary reading - while attempting to decode a model : Unexpected EOF"
    );
    let lambda: f64 = f64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap());
    offset += 8;

    assert!(
        offset + 8 < byte_stream.len(),
        "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
    );
    let number_of_layers: usize =
        u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    let mut layers: Vec<Layer> = vec![];
    for _ in 0..number_of_layers {
        let (layer, new_offset) = binary_to_layer(byte_stream, offset);
        offset = new_offset;
        layers.push(layer);
    }

    Model::init(layers, lambda, learning_step)
}
