use std::fs::read;

// 44 -> ,
// 10 -> \n
pub fn parse_test_csv() {

   let binary = read("testing_data.csv").unwrap();

   let binary_split = binary.split(|&v| v == 10 as u8);

   let a:Vec<_> = tmp.collect();
   println!("{:?}", a);

   let s = std::str::from_utf8(&res).unwrap();

   println!("{:?}", res);
   println!("{:?}", s);
}
