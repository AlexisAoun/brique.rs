use std::fs::read;

use crate::matrix::Matrix;

// 44 -> ,
// 10 -> \n
// 32 -> space
pub fn parse_test_csv() {
    let binary = read("testing_data.csv").unwrap();

    println!("{:?}", binary);

    let rows: Vec<_> = binary.split(|&v| v == 10 as u8).collect();
    let mut extracted_data: Vec<f64> = vec![];

    let mut height: usize = 0;
    let mut width: usize = 0;

    for r in rows {
        if is_line_empty(r) {
            let test: Matrix = Matrix::init(height, width, extracted_data.clone());
            test.display();

            extracted_data = vec![];
            width = 0;
            height = 0;
            continue;
        }

        if r.len() > 0 {
            println!("{:?}", tokenizer_f64(std::str::from_utf8(r).unwrap()));

            let mut tmp: Vec<f64> = tokenizer_f64(std::str::from_utf8(r).unwrap());
            if width == 0 {
                width = tmp.len();
            } else {
                assert_eq!(tmp.len(), width, "Error, not the same width");
            }
            extracted_data.append(&mut tmp);
            height += 1;
        }
    }
}

pub fn tokenizer_f64(line: &str) -> Vec<f64> {
    line.split(",")
        .filter(|s| !s.is_empty())
        .filter_map(|s| match s.parse::<f64>() {
            Ok(res) => Some(res),
            Err(e) => panic!("CSV tockenizer error, {:?}", e),
        })
        .collect::<Vec<_>>()
}

pub fn is_line_empty(line: &[u8]) -> bool {
    let mut output = true;
    for byte in line {
        if *byte != 44 as u8 && *byte != 10 as u8 && *byte != 32 as u8 {
            output = false;
            break;
        }
    }

    output
}
