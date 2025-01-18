use brique::{benchmark::benchmark, utils::generate_batch_index};

fn main() {
    //benchmark();
    //
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let m = generate_batch_index(&v, 4);

    println!("m {:?}", m);
}
