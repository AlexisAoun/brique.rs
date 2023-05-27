use crate::Matrix;

pub fn one_hot_encoding(input: &Matrix, label: &Matrix) -> Matrix {
    assert_eq!(input.height, label.width, "Input height and label width should be equal");
    let mut output = Matrix::new(1, input.height);
    for c in 0..input.height {
        output.data[0][c] = input.data[c][label.data[0][c] as usize];
    }

    output
}
