use rand::prelude::*;

// TODO use iterators
// TODO lots of optimizations left
#[derive(Clone)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
    pub transposed: bool,
}

impl Matrix {

    pub fn get(&self, row: usize, column: usize) -> f64 {
        assert!(row >= self.height, "Error while accessing matrix data : row greater or equal to height, out of bound index");
        assert!(column >= self.width, "Error while accessing matrix data : column greater or equal to width, out of bound index");

        if !self.transposed {
            self.data[row*self.width+column] 
        } else {
            self.data[column*self.height+row] 
        }
    }

    pub fn set(&mut self, value: f64, row: usize, column: usize) {
        assert!(row >= self.height, "Error while modifying matrix data : row greater or equal to height, out of bound index");
        assert!(column >= self.width, "Error while modifying matrix data : column greater or equal to width, out of bound index");
        
        if !self.transposed {
            self.data[row*self.width+column] = value;
        } else {
            self.data[column*self.height+row] = value;
        }
    }

    pub fn init_zero(height: usize, width: usize) -> Matrix {
        Matrix {
            data: vec![0.0; width*height],
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
            transposed: false
        }
    }

    pub fn init_rand(height: usize, width: usize) -> Matrix {
        let rand_vec: Vec<f64> = (0..height*width).map(|_| random::<f64>()*0.01).collect();

        Matrix {
            data: rand_vec,
            width,
            height,
            transposed: false
        }

    }

    pub fn dot(&self, m: &Matrix) -> Matrix {
        let mut res: Matrix = Matrix::init_zero(self.height, m.width);
        assert_eq!(self.width, m.height, "Error while doing a dot product: Dimension incompatibility, width of vec 1 : {}, height of vec 2 : {}", self.width, m.height);
        for c in 0..m.width {
            for r in 0..self.height {
                let mut tmp: f64 = 0.0;
                for a in 0..self.width {
                    tmp = tmp + self.get(r,a) * m.get(a,c);
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

        //let mut res: Matrix = Matrix::new(self.height, self.width);

        // for r in 0..self.height {
        //     for c in 0..self.width {
        //         res.data[r][c] = self.data[r][c] + m.data[0][c];
        //     }
        // }

        // self.data.iter().enumerate().for_each(|(index_row, row)| {
        //     row.iter().enumerate().for_each(|(index_col, value)| {
        //         res.data[index_row][index_col] = value + m.data[0][index_col]
        //     })
        // });

        // self.data.iter().enumerate().for_each(|(index, value)| {
        //     res.set(value+m.get(0, index%self.width), index/self.width, index&self.width)
        // });

        let output_vec : Vec<f64> = (0..self.height*self.width).map(|i| {
            self.data[i] + m.get(0, i%self.width)
        }).collect();

        Matrix { data: output_vec, width: self.width, height: self.height, transposed: false }
    }

    // ! IN PLACE !
    pub fn normalize(&mut self) {
        // get the maximum
        let max: f64 = *self.data.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

        self.data.iter().map(|x| x / max);
    }

    // transpose ! IN PLACE !
    pub fn transpose_inplace(&mut self) {
        self.transposed = !self.transposed;
        let tmp: usize = self.width;
        self.width = self.height;
        self.height = tmp;
    }

    pub fn t(&self) -> Matrix {
        let output = self.clone();
        output.transpose_inplace();
        output
    }

    // used for test
    pub fn is_equal(&self, m: &Matrix, precision: i32) -> bool {
        if self.width != m.width || self.height != m.height {
            return false;
        } else {
            for i in 0..self.height*self.width {
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
        self.data.iter().map(|x| x.exp());
    }

    pub fn exp(&self) -> Matrix {
        let mut output = self.clone();
        output.exp_inplace();
        output
    }

    pub fn pow_inplace(&mut self, a: i32) {
        self.data.iter().map(|x| x.powi(a));
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
        // for c in 0..self.width {
        //     for r in 0..self.height {
        //         output.data[0][c] += self.data[r][c];
        //     }
        // }

        self.data.iter().enumerate().for_each(|(index, value)| {
            output.data[index%self.width] += value
        });

        output
    }

    pub fn div_inplace(&mut self, a: f64) {
        assert_ne!(a, 0.0, "Divide by 0 matrix error");
        self.data.iter().map(|x| x/a);
    }

    pub fn div(&self, a: f64) -> Matrix {
        let mut output: Matrix = self.clone();
        output.div_inplace(a);
        output
    }

    pub fn mult_inplace(&mut self, a: f64) {
        self.data.iter().map(|x| x*a);
    }

    pub fn mult(&self, a: f64) -> Matrix {
        let mut output: Matrix = self.clone();
        output.mult_inplace(a);
        output
    }

    //TODO inplace
    pub fn add_two_matrices(&self, m: &Matrix) -> Matrix {
        assert!(
            self.height == m.height && self.width == m.width,
            "The two matrices should have the same dimensions"
        );
        let output_vec : Vec<f64> = (0..self.height*self.width).map(|i| self.data[i]+m.data[i]).collect();
        
        Matrix { data: output_vec, width: self.width, height: self.height, transposed: false }
    }

    pub fn display(&self) {
        print!("\n");
        print!("-------------");
        print!("\n");
        for i in 0..self.height {
            for j in 0..self.width {
                print!(" {} |", self.get(i,j));
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
                output.push_str(&self.get(i,j).to_string());
                output.push(',');
            }
            output.push('\n');
        }

        output
    }
}
