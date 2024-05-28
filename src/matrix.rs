// To acces Matrix data : Matrix.data[row][column]
use rand::prelude::*;

// TODO generic type for matrix data
// TODO use iterators
#[derive(Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub width: usize,
    pub height: usize,
}

impl Matrix {
    pub fn new(height: usize, width: usize) -> Matrix {
        Matrix {
            data: vec![vec![0.0; width]; height],
            width,
            height,
        }
    }

    pub fn init(height: usize, width: usize, data: Vec<f64>) -> Matrix {
        assert_eq!(
            height * width,
            data.len(),
            "Error while initiating a matrix with data : 
                   not compatible with the dimension"
        );

        let mut output = Self::new(height, width);

        let mut index: usize = 0;
        for i in data {
            let c: usize = index % width;
            let r: usize = index / width;

            output.data[r][c] = i;
            index += 1;
        }

        output
    }

    pub fn init_rand(height: usize, width: usize) -> Matrix {
        let mut output = Self::new(height, width);

        for r in 0..height {
            for c in 0..width {
                let x: f64 = random();
                output.data[r][c] = x * 0.01;
            }
        }
        output
    }

    pub fn dot(&self, m: &Matrix) -> Matrix {
        let mut res: Matrix = Matrix::new(self.height, m.width);
        if self.width == m.height {
            for c in 0..m.width {
                for r in 0..self.height {
                    let mut tmp: f64 = 0.0;
                    for a in 0..self.width {
                        tmp = tmp + self.data[r][a] * m.data[a][c];
                    }
                    res.data[r][c] = tmp;
                }
            }
        } else {
            panic!("Error while doing a dot product: Dimension incompatibility")
        }
        res
    }

    // adds a matrix of X width and 1 height to a matrix of Y height and X width
    pub fn add_value_to_all_rows(&self, m: &Matrix) -> Matrix {
        assert_eq!(m.height, 1, "The input matrix should have a height of 1");
        assert_eq!(
            m.width, self.width,
            "The 2 matrices should have the same width"
        );
        let mut res: Matrix = Matrix::new(self.height, self.width);

        for r in 0..self.height {
            for c in 0..self.width {
                res.data[r][c] = self.data[r][c] + m.data[0][c];
            }
        }

        res
    }

    pub fn normalize(&self) -> Matrix {
        let mut output: Matrix = Matrix::new(self.height, self.width);

        // get the maximum
        let mut max: f64 = 0.0;
        for r in 0..self.height {
            for c in 0..self.width {
                if self.data[r][c] > max {
                    max = self.data[r][c];
                }
            }
        }

        // normalize
        for r in 0..self.height {
            for c in 0..self.width {
                output.data[r][c] = self.data[r][c] / max;
            }
        }

        output
    }

    // transpose
    pub fn t(&self) -> Matrix {
        let mut output: Matrix = Matrix::new(self.width, self.height);

        for r in 0..output.height {
            for c in 0..output.width {
                output.data[r][c] = self.data[c][r];
            }
        }
        output
    }

    // used for test
    pub fn is_equal(&self, m: &Matrix, precision: i32) -> bool {
        if self.width != m.width || self.height != m.height {
            return false;
        } else {
            for r in 0..self.height {
                for c in 0..self.width {
                    let mut a: f64 = self.data[r][c] * 10_f64.powi(precision);
                    a = a.round() / 10_f64.powi(precision);

                    let mut b: f64 = m.data[r][c] * 10_f64.powi(precision);
                    b = b.round() / 10_f64.powi(precision);

                    if a != b {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn exp(&self) -> Matrix {
        let mut output: Matrix = Matrix::new(self.height, self.width);
        for r in 0..self.height {
            for c in 0..self.width {
                output.data[r][c] = self.data[r][c].exp();
            }
        }
        output
    }

    pub fn pow(&self, a: i32) -> Matrix {
        let mut output: Matrix = Matrix::new(self.height, self.width);
        for r in 0..self.height {
            for c in 0..self.width {
                output.data[r][c] = self.data[r][c].powi(a);
            }
        }
        output
    }

    pub fn sum(&self) -> f64 {
        let mut sum: f64 = 0.0;
        for r in 0..self.height {
            for c in 0..self.width {
                sum += self.data[r][c];
            }
        }
        sum
    }

    pub fn sum_rows(&self) -> Matrix {
        let mut output: Matrix = Matrix::new(1, self.width);
        for c in 0..self.width {
            for r in 0..self.height {
                output.data[0][c] += self.data[r][c];
            }
        }

        output
    }

    pub fn div(&self, value: f64) -> Matrix {
        assert_ne!(value, 0.0, "Divide by 0 matrix error");
        let mut output: Matrix = Matrix::new(self.height, self.width);
        for r in 0..self.height {
            for c in 0..self.width {
                output.data[r][c] = self.data[r][c] / value;
            }
        }
        output
    }

    pub fn mult(&self, value: f64) -> Matrix {
        let mut output: Matrix = Matrix::new(self.height, self.width);
        for r in 0..self.height {
            for c in 0..self.width {
                output.data[r][c] = self.data[r][c] * value;
            }
        }
        output
    }

    pub fn add_two_matrices(&self, m: &Matrix) -> Matrix {
        assert!(
            self.height == m.height && self.width == m.width,
            "The two matrices should have the same dimensions"
        );
        let mut res: Matrix = Matrix::new(self.height, self.width);

        for r in 0..self.height {
            for c in 0..self.width {
                res.data[r][c] = self.data[r][c] + m.data[r][c];
            }
        }

        res
    }

    pub fn display(&self) {
        print!("\n");
        print!("-------------");
        print!("\n");
        for i in 0..self.height {
            for j in 0..self.width {
                print!(" {} |", self.data[i][j]);
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
                output.push_str(&self.data[i][j].to_string());
                output.push(',');
            }
            output.push('\n');
        }

        output
    }
}
