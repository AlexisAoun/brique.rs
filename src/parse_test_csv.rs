use std::fs::read;

// 44 -> ,
// 10 -> \n
pub fn parse_test_csv() {

   let binary = read("testing_data.csv").unwrap();

   let rows : Vec<_> = binary.split(|&v| v == 10 as u8).collect();

   for r in rows {
       if r.len() > 0 {
            println!("{:?}", tokenizer_f64(std::str::from_utf8(r).unwrap()));
       }
   } 

}

pub fn tokenizer_f64(line : &str) -> Vec<f64> {
    line.split(",").filter_map(|s| match s.parse::<f64>() {
        Ok(res) => Some(res),
        Err(e) => panic!("CSV tockenizer error, {:?}", e)
    }).collect::<Vec<_>>()
}
