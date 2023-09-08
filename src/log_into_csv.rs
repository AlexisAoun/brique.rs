use std::{
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
};

use crate::matrix::Matrix;

pub fn log_matrix_into_csv(title: &str, matrix: &Matrix) {
    let mut file: File = match OpenOptions::new()
        .write(true)
        .append(true)
        .open("logs.csv") {
            Ok(file) => file,
            Err(err) => match err.kind() {
                std::io::ErrorKind::NotFound => match File::create("logs.csv") {
                    Ok(file) => file,
                    Err(err) => panic!("Cannot create csv logging file : {}", err)
                }
                other_error => panic!("Cannot open csv logging file : {}", other_error)
            }
        };

    file.seek(SeekFrom::End(0)).unwrap();
    let title_to_write = format!("{}\n", title);
    let matrix_to_write = matrix.convert_to_csv();

    file.write(title_to_write.as_bytes()).unwrap();
    file.write(matrix_to_write.as_bytes()).unwrap();
}
