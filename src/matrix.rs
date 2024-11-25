use rand::prelude::*;

#[derive(Clone)]
pub struct Matrix {
    data: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub transposed: bool,
}

impl Matrix {
    pub fn init_zero(height: usize, width: usize) -> Matrix {
        Matrix {
            data: vec![0.0; width * height],
            width,
            height,
            transposed: false,
        }
    }

    pub fn init(height: usize, width: usize, data: Vec<f64>) -> Matrix {
        assert_eq!(
            height * width,
            data.len(),
            "Error while initiating a matrix with data : 
                   not compatible with the dimension"
        );

        Matrix {
            data,
            width,
            height,
            transposed: false,
        }
    }

    pub fn init_rand(height: usize, width: usize) -> Matrix {
        let rand_vec: Vec<f64> = (0..height * width)
            .map(|_| random::<f64>() * 0.01)
            .collect();

        Matrix {
            data: rand_vec,
            width,
            height,
            transposed: false,
        }
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        assert!(row < self.height, "Error while accessing matrix data : row greater or equal to height, out of bound index");
        assert!(column < self.width, "Error while accessing matrix data : column greater or equal to width, out of bound index");

        if !self.transposed {
            self.data[row * self.width + column]
        } else {
            self.data[column * self.height + row]
        }
    }

    // access to underlying one dimensional Vec
    pub fn get_1d(&self, index: usize) -> f64 {
        assert!(
            index < self.data.len(),
            "Error while accessing matrix data : index greater than vec size, out of bound index"
        );

        self.data[index]
    }

    pub fn get_row(&self, row: usize) -> Vec<f64> {
        assert!(row < self.height, "Error while accessing matrix data : row greater or equal to height, out of bound index");

        let mut output: Vec<f64> = Vec::new();

        for i in 0..self.width {
            output.push(self.get(row, i));
        }

        output
    }

    pub fn set(&mut self, value: f64, row: usize, column: usize) {
        assert!(row < self.height, "Error while modifying matrix data : row greater or equal to height, out of bound index");
        assert!(column < self.width, "Error while modifying matrix data : column greater or equal to width, out of bound index");

        if !self.transposed {
            self.data[row * self.width + column] = value;
        } else {
            self.data[column * self.height + row] = value;
        }
    }

    // access to underlying one dimensional Vec
    pub fn set_1d(&mut self, value: f64, index: usize) {
        assert!(
            index < self.data.len(),
            "Error while accessing matrix data : index greater than vec size, out of bound index"
        );

        self.data[index] = value;
    }

    pub fn set_row(&mut self, new_row: &Vec<f64>, row: usize) {
        assert!(row < self.height, "Error while accessing matrix data : row greater or equal to height, out of bound index");

        for i in 0..self.width {
            self.set(new_row[i], row, i);
        }
    }

    pub fn dot(&self, m: &Matrix) -> Matrix {
        let mut res: Matrix = Matrix::init_zero(self.height, m.width);
        assert_eq!(self.width, m.height, "Error while doing a dot product: Dimension incompatibility, width of vec 1 : {}, height of vec 2 : {}", self.width, m.height);
        for c in 0..m.width {
            for r in 0..self.height {
                let mut tmp: f64 = 0.0;
                for a in 0..self.width {
                    tmp = tmp + self.get(r, a) * m.get(a, c);
                }
                res.set(tmp, r, c);
            }
        }
        res
    }

    // adds a matrix of X width and 1 height to a matrix of Y height and X width
    pub fn add_1d_matrix_to_all_rows(&self, m: &Matrix) -> Matrix {
        assert_eq!(m.height, 1, "The input matrix should have a height of 1");
        assert_eq!(
            m.width, self.width,
            "The 2 matrices should have the same width"
        );

        let output_vec: Vec<f64> = (0..self.height * self.width)
            .map(|i| self.data[i] + m.get(0, i % self.width))
            .collect();

        Matrix {
            data: output_vec,
            width: self.width,
            height: self.height,
            transposed: false,
        }
    }

    pub fn max(&self) -> f64 {
        *self.data.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
    }

    pub fn min(&self) -> f64 {
        *self.data.iter().min_by(|a, b| a.total_cmp(b)).unwrap()
    }

    // ! IN PLACE !
    pub fn normalize(&mut self) {
        // get the maximum
        let max: f64 = self.max();
        let min: f64 = self.min();

        self.data = self.data.iter().map(|x| (x - min) / (max - min)).collect();
    }

    // transpose ! IN PLACE !
    pub fn transpose_inplace(&mut self) {
        self.transposed = !self.transposed;
        let tmp: usize = self.width;
        self.width = self.height;
        self.height = tmp;
    }

    pub fn t(&self) -> Matrix {
        let mut output = self.clone();
        output.transpose_inplace();
        output
    }

    // used for test
    pub fn is_equal(&self, m: &Matrix, precision: i32) -> bool {
        if self.width != m.width || self.height != m.height {
            return false;
        } else {
            for i in 0..self.height * self.width {
                let mut a: f64 = self.data[i] * 10_f64.powi(precision);
                a = a.round() / 10_f64.powi(precision);

                let mut b: f64 = m.data[i] * 10_f64.powi(precision);
                b = b.round() / 10_f64.powi(precision);

                if a != b {
                    return false;
                }
            }
        }
        true
    }

    pub fn exp_inplace(&mut self) {
        self.data = self.data.iter().map(|x| x.exp()).collect();
    }

    pub fn exp(&self) -> Matrix {
        let mut output = self.clone();
        output.exp_inplace();
        output
    }

    pub fn pow_inplace(&mut self, a: i32) {
        self.data = self.data.iter().map(|x| x.powi(a)).collect();
    }

    pub fn pow(&self, a: i32) -> Matrix {
        let mut output: Matrix = self.clone();
        output.pow_inplace(a);

        output
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn sum_rows(&self) -> Matrix {
        let mut output: Matrix = Matrix::init_zero(1, self.width);

        self.data
            .iter()
            .enumerate()
            .for_each(|(index, value)| output.data[index % self.width] += value);

        output
    }

    pub fn div_inplace(&mut self, a: f64) {
        assert_ne!(a, 0.0, "Divide by 0 matrix error");
        self.data = self.data.iter().map(|x| x / a).collect();
    }

    pub fn div(&self, a: f64) -> Matrix {
        let mut output: Matrix = self.clone();
        output.div_inplace(a);
        output
    }

    pub fn mult_inplace(&mut self, a: f64) {
        self.data = self.data.iter().map(|x| x * a).collect();
    }

    pub fn mult(&self, a: f64) -> Matrix {
        let mut output: Matrix = self.clone();
        output.mult_inplace(a);
        output
    }

    // zip iterators the two arrays
    pub fn add_two_matrices(&self, m: &Matrix) -> Matrix {
        assert!(
            self.height == m.height && self.width == m.width,
            "The two matrices should have the same dimensions"
        );
        let output_vec: Vec<f64> = (0..self.height * self.width)
            .map(|i| self.data[i] + m.data[i])
            .collect();

        Matrix {
            data: output_vec,
            width: self.width,
            height: self.height,
            transposed: false,
        }
    }

    pub fn pop_last_row(&mut self) {
        let begin_index = self.height * (self.width - 1);
        let last_index = self.height * self.width;

        for _i in begin_index..last_index {
            self.data.pop();
        }

        self.height -= 1;
    }

    pub fn compute_d_relu_inplace(&mut self, z_minus_1: &Matrix) {
        self.data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| if z_minus_1.data[i] <= 0.0 { 0.0 } else { *v })
            .collect();
    }

    pub fn display(&self) {
        print!("\n");
        print!("-------------");
        print!("\n");
        for i in 0..self.height {
            for j in 0..self.width {
                print!(" {} |", self.get(i, j));
            }
            print!("/ \n");
        }
        print!("-------------");
        print!("\n");
    }

    pub fn convert_to_csv(&self) -> String {
        let mut output: String = String::new();
        for i in 0..self.height {
            for j in 0..self.width {
                output.push_str(&self.get(i, j).to_string());
                output.push(',');
            }
            output.push('\n');
        }

        output
    }
}

//unit test
#[cfg(test)]
mod tests {
    use crate::parse_test_csv::parse_test_csv;

    use super::Matrix;

    fn get_test_matrix() -> Matrix {
        let matrix = Matrix::init(2, 3, vec![0.1, 1.3, 0.5, 12.0, 1.01, -1000.0]);

        matrix
    }

    #[test]
    fn valid_get() {
        let matrix = get_test_matrix();

        assert_eq!(matrix.get(0, 0), 0.1);
        assert_eq!(matrix.get(0, 1), 1.3);
        assert_eq!(matrix.get(0, 2), 0.5);
        assert_eq!(matrix.get(1, 0), 12.0);
        assert_eq!(matrix.get(1, 1), 1.01);
        assert_eq!(matrix.get(1, 2), -1000.0);
    }

    #[test]
    fn valid_get_on_transposed() {
        let mut matrix = get_test_matrix();
        matrix.transpose_inplace();

        assert_eq!(matrix.get(0, 0), 0.1);
        assert_eq!(matrix.get(0, 1), 12.0);
        assert_eq!(matrix.get(1, 0), 1.3);
        assert_eq!(matrix.get(1, 1), 1.01);
        assert_eq!(matrix.get(2, 0), 0.5);
        assert_eq!(matrix.get(2, 1), -1000.0);
    }

    #[test]
    fn valid_get_on_untransposed() {
        let mut matrix = get_test_matrix();
        matrix.transpose_inplace();
        matrix.transpose_inplace();

        assert_eq!(matrix.get(0, 0), 0.1);
        assert_eq!(matrix.get(0, 1), 1.3);
        assert_eq!(matrix.get(0, 2), 0.5);
        assert_eq!(matrix.get(1, 0), 12.0);
        assert_eq!(matrix.get(1, 1), 1.01);
        assert_eq!(matrix.get(1, 2), -1000.0);
    }

    #[test]
    #[should_panic]
    fn unvalid_get_column_out_of_bound() {
        let matrix = get_test_matrix();

        matrix.get(2, 0);
    }

    #[test]
    #[should_panic]
    fn unvalid_get_row_out_of_bound() {
        let matrix = get_test_matrix();

        matrix.get(5, 1);
    }

    #[test]
    #[should_panic]
    fn unvalid_get_tranposed_column_out_of_bound() {
        let mut matrix = get_test_matrix();
        matrix.transpose_inplace();

        matrix.get(0, 2);
    }

    #[test]
    #[should_panic]
    fn unvalid_get_transposed_row_out_of_bound() {
        let mut matrix = get_test_matrix();
        matrix.transpose_inplace();

        matrix.get(3, 1);
    }

    #[test]
    fn valid_get_row() {
        let matrix = get_test_matrix();
        let expected_vec = vec![12.0, 1.01, -1000.0];

        assert_eq![matrix.get_row(1), expected_vec];
    }

    #[test]
    fn valid_get_row_on_transposed() {
        let mut matrix = get_test_matrix();
        let expected_vec = vec![0.5, -1000.0];
        matrix.transpose_inplace();

        assert_eq![matrix.get_row(2), expected_vec];
    }

    #[test]
    fn valid_set() {
        let mut matrix = get_test_matrix();
        matrix.set(69.69, 1, 1);

        assert_eq![matrix.data[4], 69.69];
    }

    #[test]
    #[should_panic]
    fn unvalid_set_column_out_of_bound() {
        let mut matrix = get_test_matrix();

        matrix.set(69.69, 2, 0);
    }

    #[test]
    #[should_panic]
    fn unvalid_set_row_out_of_bound() {
        let mut matrix = get_test_matrix();

        matrix.set(69.69, 5, 1);
    }

    #[test]
    #[should_panic]
    fn unvalid_set_tranposed_column_out_of_bound() {
        let mut matrix = get_test_matrix();
        matrix.transpose_inplace();

        matrix.set(69.69, 0, 2);
    }

    #[test]
    #[should_panic]
    fn unvalid_set_transposed_row_out_of_bound() {
        let mut matrix = get_test_matrix();
        matrix.transpose_inplace();

        matrix.set(69.69, 3, 1);
    }

    #[test]
    fn valid_set_row() {
        let mut matrix = get_test_matrix();
        let new_row = vec![0.8, 0.1, 1203123.0];

        matrix.set_row(&new_row, 0);

        assert_eq![matrix.get_row(0), new_row];
    }

    #[test]
    fn valid_set_row_on_transposed() {
        let mut matrix = get_test_matrix();
        let new_row = vec![0.8, 0.1];

        matrix.transpose_inplace();
        matrix.set_row(&new_row, 2);

        assert_eq![matrix.get_row(2), new_row];
    }

    #[test]
    fn max_test() {
        let matrix = get_test_matrix();

        assert_eq![matrix.max(), 12.0];
    }

    #[test]
    fn min_test() {
        let matrix = get_test_matrix();

        assert_eq![matrix.min(), -1000.0];
    }

    #[test]
    fn add_values_of_a_row_test() {
        let test_data = parse_test_csv("tests/test_data/add_values_of_a_row_test.csv".to_string());

        assert!(test_data[0]
            .add_1d_matrix_to_all_rows(&test_data[1])
            .is_equal(&test_data[2], 10));
    }

    #[test]
    fn dot_product_test() {
        let test_data = parse_test_csv("tests/test_data/dot_product_test.csv".to_string());

        assert!(test_data[0].dot(&test_data[1]).is_equal(&test_data[2], 8));
    }

    #[test]
    fn normalize_test() {
        let mut test_data = parse_test_csv("tests/test_data/normalize_test.csv".to_string());

        test_data[0].normalize();

        assert!(test_data[0].is_equal(&test_data[1], 8));
    }
}
