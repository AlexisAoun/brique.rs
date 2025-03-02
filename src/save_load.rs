use crate::{layers::Layer, matrix::Matrix, model::Model, optimizer::Optimizer};
use core::panic;
use std::{collections::HashMap, fmt, fs};

const FILE_EXTENSION: &str = ".brq";
const VERSION: u8 = 2;
const HEADER_SIZE: u64 = 15;
// CAT
const START_OF_OBJECT_MAGIC_NUMBER: [u8; 3] = [67, 65, 84];
// COOKIE
const START_OF_FILE_MAGIC_NUMBER: [u8; 6] = [67, 79, 79, 75, 73, 69];

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

#[derive(Debug)]
pub enum ModelManagementError {
    CouldNotSaveModel(String),
    CouldNotReadFile(String),
    CouldNotDecodeBinary(String),
}

impl fmt::Display for ModelManagementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelManagementError::CouldNotSaveModel(msg) => {
                write!(f, "Could not save the model, details : {}", msg)
            }
            ModelManagementError::CouldNotReadFile(msg) => {
                write!(f, "Could not read file, details : {}", msg)
            }
            ModelManagementError::CouldNotDecodeBinary(msg) => {
                write!(f, "Could not decode binary, details : {}", msg)
            }
        }
    }
}

pub fn save_model(model: &Model, file_path: String) -> Result<(), ModelManagementError> {
    let mut byte_stream: Vec<u8> = vec![];
    byte_stream.append(&mut model_to_binary(model));
    byte_stream.splice(0..0, add_header(byte_stream.len() as u64));

    let res_write = fs::write(file_path + FILE_EXTENSION, byte_stream);

    if res_write.is_ok() {
        Ok(())
    } else {
        Err(ModelManagementError::CouldNotSaveModel(
            res_write.unwrap_err().to_string(),
        ))
    }
}

pub fn load_model(file_path: String) -> Result<Model, ModelManagementError> {
    let byte_stream: Vec<u8> = match fs::read(file_path + FILE_EXTENSION) {
        Ok(output) => output,
        Err(e) => return Err(ModelManagementError::CouldNotReadFile(e.to_string())),
    };

    match check_header(&byte_stream) {
        Ok(()) => (),
        Err(e) => return Err(e),
    };

    binary_to_model(&byte_stream, HEADER_SIZE as usize)
}

// header (size 15 bytes)
// magic number : 6 bytes
// version, would match the version of the release of the lib, i.e, 0.2 => 2, 0.3 => 3 .... 1.0 => 10, 1.1 => 11 : 1 byte
// length of the binary (data and header combined) in bytes : 8 bytes
pub fn add_header(data_size: u64) -> Vec<u8> {
    let mut header: Vec<u8> = vec![];
    header.append(&mut START_OF_FILE_MAGIC_NUMBER.to_vec());
    header.push(VERSION);
    header.append(&mut (data_size + HEADER_SIZE).to_be_bytes().to_vec());

    header
}

pub fn check_header(byte_stream: &Vec<u8>) -> Result<(), ModelManagementError> {
    let mut offset: usize = 0;
    if offset + 6 > byte_stream.len() {
        return Err(ModelManagementError::CouldNotDecodeBinary(
            "while attempting to decode the header : Unexpected EOF".to_string(),
        ));
    }
    if byte_stream[offset..offset + 6] != START_OF_FILE_MAGIC_NUMBER {
        return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode the header : Binary start of the file code not found, file may be corrupted".to_string()));
    }

    offset += 6;

    if offset > byte_stream.len() {
        return Err(ModelManagementError::CouldNotDecodeBinary(
            "while attempting to decode the header : Unexpected EOF".to_string(),
        ));
    }

    if byte_stream[offset] != VERSION {
        return Err(ModelManagementError::CouldNotDecodeBinary(
            "while attempting to decode the header : wrong file version".to_string(),
        ));
    }
    offset += 1;

    if offset + 8 > byte_stream.len() {
        return Err(ModelManagementError::CouldNotDecodeBinary(
            "while attempting to decode the header : Unexpected EOF".to_string(),
        ));
    }
    let length: u64 = u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap());

    if length != byte_stream.len() as u64 {
        return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode the header : file length different than expected, file may be corrupted".to_string()));
    }

    Ok(())
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

pub fn binary_to_matrix(
    byte_stream: &Vec<u8>,
    input_offset: usize,
) -> Result<(Matrix, usize), ModelManagementError> {
    let mut offset = input_offset;

    if byte_stream[offset..offset + 3] != START_OF_OBJECT_MAGIC_NUMBER {
        return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode a matrix : Binary start of object code not found, file may be corrupted".to_string()));
    }

    offset += 3;
    let id_lookup_table = LookupStructBinaryId::init();

    if byte_stream[offset] != id_lookup_table.lookup("Matrix") {
        return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode a matrix : Binary id code does not match the lookup table for the Matrix entry, file may be corrupted".to_string()));
    }
    offset += 1;

    let transposed: bool = byte_stream[offset] != 0;
    offset += 1;

    let height: usize =
        u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    let width: usize =
        u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    let data_size: usize = height * width * 8;
    if offset + data_size > byte_stream.len() {
        return Err(ModelManagementError::CouldNotDecodeBinary(
            "Save binary reading - while attempting to decode a matrix : Unexpected EOF"
                .to_string(),
        ));
    }

    if offset + data_size + 3 <= byte_stream.len() {
        if byte_stream[offset + data_size..offset + data_size + 3] != START_OF_OBJECT_MAGIC_NUMBER {
            return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode a matrix : Binary start of object code not found, file may be corrupted".to_string()));
        }
    } else {
        if offset + data_size != byte_stream.len() {
            return Err(ModelManagementError::CouldNotDecodeBinary(
                "while attempting to decode a matrix : Unexpected EOF".to_string(),
            ));
        }
    }

    let data: Vec<f64> = binary_to_f64_array(byte_stream[offset..offset + data_size].to_vec());
    offset += data_size;

    let output_matrix = Matrix {
        height,
        width,
        transposed,
        data,
    };

    Ok((output_matrix, offset))
}

// weights : matrix
// biases : matrix
// activation : bool
pub fn layer_to_binary(input_layer: &Layer) -> Vec<u8> {
    let mut output: Vec<u8> = vec![];

    let id_lookup_table = LookupStructBinaryId::init();

    output.append(&mut START_OF_OBJECT_MAGIC_NUMBER.to_vec());
    output.push(id_lookup_table.lookup("Layer"));
    output.push(input_layer.relu as u8);
    output.append(&mut matrix_to_binary(&input_layer.weights_t));
    output.append(&mut matrix_to_binary(&input_layer.biases));

    output
}

pub fn binary_to_layer(
    byte_stream: &Vec<u8>,
    input_offset: usize,
) -> Result<(Layer, usize), ModelManagementError> {
    let mut offset = input_offset;

    if byte_stream[offset..offset + 3] != START_OF_OBJECT_MAGIC_NUMBER {
        return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode a layer : Binary start of object code not found, file may be corrupted".to_string()));
    }
    offset += 3;
    let id_lookup_table = LookupStructBinaryId::init();

    if byte_stream[offset] != id_lookup_table.lookup("Layer") {
        return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode a layer : Binary id code does not match the lookup table for the Matrix entry, file may be corrupted".to_string()));
    }
    offset += 1;

    let activation: bool = byte_stream[offset] != 0;
    offset += 1;

    let (weights_t, offset) = match binary_to_matrix(byte_stream, offset) {
        Ok((matrix, offset)) => (matrix, offset),
        Err(e) => return Err(e),
    };

    let (biases, offset) = match binary_to_matrix(byte_stream, offset) {
        Ok((matrix, offset)) => (matrix, offset),
        Err(e) => return Err(e),
    };

    let output_layer = Layer::init_with_data(weights_t, biases, activation);

    Ok((output_layer, offset))
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
    output.append(&mut input_model.lambda.to_be_bytes().to_vec());
    output.append(&mut (input_model.layers.len() as u64).to_be_bytes().to_vec());

    input_model
        .layers
        .iter()
        .for_each(|layer| output.append(&mut layer_to_binary(&layer)));

    output
}

pub fn binary_to_model(
    byte_stream: &Vec<u8>,
    input_offset: usize,
) -> Result<Model, ModelManagementError> {
    let mut offset: usize = input_offset;

    if byte_stream[offset..offset + 3] != START_OF_OBJECT_MAGIC_NUMBER {
        return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode the model : Binary start of object code not found, file may be corrupted".to_string()));
    }

    offset += 3;
    let id_lookup_table = LookupStructBinaryId::init();

    if byte_stream[offset] != id_lookup_table.lookup("Model") {
        return Err(ModelManagementError::CouldNotDecodeBinary("while attempting to decode the model : Binary id code does not match the lookup table for the Matrix entry, file may be corrupted".to_string()));
    }
    offset += 1;

    let lambda: f64 = f64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap());
    offset += 8;

    let number_of_layers: usize =
        u64::from_be_bytes(byte_stream[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    let mut layers: Vec<Layer> = vec![];
    for _ in 0..number_of_layers {
        let (layer, new_offset) = match binary_to_layer(byte_stream, offset) {
            Ok((layer, offset)) => (layer, offset),
            Err(e) => return Err(e),
        };

        offset = new_offset;
        layers.push(layer);
    }

    Ok(Model::init(
        layers,
        Optimizer::SGD {
            learning_step: 0.01,
        },
        lambda,
    ))
}

//unit test
#[cfg(test)]
mod tests {
    use core::panic;
    use std::fs;

    use crate::{layers::Layer, model::Model, optimizer::Optimizer, save_load::FILE_EXTENSION};

    use super::{load_model, save_model};

    #[test]
    fn succesful_model_save_and_load() {
        let layer1 = Layer::init(10, 100, true);
        let layer2 = Layer::init(100, 200, true);
        let layer3 = Layer::init(200, 200, true);
        let layer4 = Layer::init(200, 3, false);

        let lambda: f64 = 0.012;

        let file_path: String = "test_model_save".to_string();
        let model = Model::init(
            vec![layer1, layer2, layer3, layer4],
            Optimizer::SGD {
                learning_step: 0.01,
            },
            lambda,
        );
        save_model(&model, file_path.clone()).unwrap();

        let loaded_model = match load_model(file_path.clone()) {
            Ok(model) => model,
            Err(e) => panic!("{}", e),
        };

        match fs::remove_file(file_path + FILE_EXTENSION) {
            Ok(()) => (),
            Err(e) => panic!("{}", e),
        };

        assert_eq!(
            model.lambda, loaded_model.lambda,
            "Models lambdas are not the same"
        );
        assert_eq!(
            model.layers.len(),
            loaded_model.layers.len(),
            "Models do not have the same number of layers"
        );
        for i in 0..model.layers.len() {
            assert!(
                model.layers[i]
                    .weights_t
                    .is_equal(&loaded_model.layers[i].weights_t, 10),
                "Layer {} weights are different in the two models",
                i
            );
            assert!(
                model.layers[i]
                    .biases
                    .is_equal(&loaded_model.layers[i].biases, 10),
                "Layer {} biases are different in the two models",
                i
            );
        }
    }
}
