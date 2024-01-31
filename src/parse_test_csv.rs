use std::fs::read;

use crate::matrix::Matrix;

// 44 -> ,
// 10 -> \n
pub fn parse_test_csv() {

   let binary = read("testing_data.csv").unwrap();

   let rows : Vec<_> = binary.split(|&v| v == 10 as u8).collect();
   let mut extracted_data: Vec<f64> = vec![];

   let mut height:usize = 0;
   let mut width:usize = 0;

   for r in rows {
       if r.len() > 0 {
            println!("{:?}", tokenizer_f64(std::str::from_utf8(r).unwrap()));

            let mut tmp:Vec<f64> = tokenizer_f64(std::str::from_utf8(r).unwrap());
            if width == 0 {
                width = tmp.len();
            } else {
                assert_eq!(tmp.len(), width, "Error, not the same width");
            }
            extracted_data.append(&mut tmp);    
            height+=1;
       }
   } 

   let test:Matrix = Matrix::init(height, width, extracted_data);

   test.display();

}

pub fn tokenizer_f64(line : &str) -> Vec<f64> {
    line.split(",").filter_map(|s| match s.parse::<f64>() {
        Ok(res) => Some(res),
        Err(e) => panic!("CSV tockenizer error, {:?}", e)
    }).collect::<Vec<_>>()
}
