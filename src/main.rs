mod matrice;

use crate::matrice::Matrice;
use std::fs::read;

fn convert_4_bytes_to_u32_big_endian(bytes: Vec<u8>) -> u32 {

    assert_eq!(bytes.len(), 4, "byte array should be of size 4");
    let output: u32 = (bytes[0] as u32) * 2_u32.pow(24)
        + (bytes[1] as u32) * 2_u32.pow(16)
        + (bytes[2] as u32) * 2_u32.pow(8)
        + (bytes[3] as u32);

    output
}

fn extract_labels(path: &str) -> Vec<u8> {
    let res: Vec<u8> = read(path).unwrap();
    assert_eq!(res[0..4].to_vec(), [0,0,8,1], "File incompatibility detected, are you sure you added the correct LABEL file ?");

    let array_size: u32 = convert_4_bytes_to_u32_big_endian(res[4..8].to_vec());
    assert_eq!(array_size, res.len() as u32 - 8,"File incompatibility detected, are you sure you added the correct LABEL file ?");

    res[8..].to_vec()
}

fn extract_images(path: &str) -> Matrice {
    let res: Vec<u8> = read(path).unwrap();
    assert_eq!(res[0..4].to_vec(), [0,0,8,3], "File incompatibility detected, are you sure you added the correct IMAGE file ?");

    let array_size: u32 = convert_4_bytes_to_u32_big_endian(res[4..8].to_vec());
    let array_size_row: u32 = convert_4_bytes_to_u32_big_endian(res[8..12].to_vec());
    let array_size_column: u32 = convert_4_bytes_to_u32_big_endian(res[12..16].to_vec());
    assert_eq!(array_size*array_size_column*array_size_row, res.len() as u32 - 16,"File incompatibility detected, are you sure you added the correct IMAGE file ?");

    let mut output: Matrice = Matrice::new(array_size, array_size_row*array_size_column);
    let mut index = 0;

    println!("{}", res[16..].to_vec().len()/(28*28));
    for i in res[16..].to_vec() {
        output.data[index/(28*28)][index%(28*28)] = i as f64;
        index+=1;
    }

    output
}

fn main() {
    let labels: Vec<u8> = extract_labels("data/train-labels.idx1-ubyte");
    let images: Matrice = extract_images("data/train-images.idx3-ubyte");

    println!("{}", labels[8]);

    for i in 0..28*28 {
        if images.data[8][i] > 0f64 {
            if i%28 == 0 { print!("\n"); }
            print!("*");    
        } else {
            if i%28 == 0 { print!("\n"); }
            print!("_");
        }
    }


    // matrice test 
    // let mut m1: Matrice = Matrice::new(2,2);
    // m1.data[0][0] = -3242.213;
    // m1.data[1][0] = 1242356.4245;
    // m1.data[0][1] = 41466.9088;
    // m1.data[1][1] = 0.0;
    // let mut m2: Matrice = Matrice::new(2,3);
    // m2.data[0][0] = 898.0;
    // m2.data[1][0] = -222.3;
    // m2.data[0][1] = -2467.9098;
    // m2.data[1][1] = 1356770.0;
    // m2.data[0][2] = -696969.69696;
    // m2.data[1][2] = 10.0;
    // m1.display();
    // m2.display();
    //
    // let m3 = m2.dot(m1);
    // m3.display();

}
