// To acces Matrix data : Matrix.data[row][column]
use rand::prelude::*;

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

    pub fn init_rand(height: usize, width: usize) -> Matrix {
        let mut output = Self::new(height, width);

        for r in 0..height {
            for c in 0..width {
                let x: f64 = random();
                output.data[r][c] = x*0.01;
            }
        }
        output
    }

    pub fn dot(&self, m: &Matrix) -> Matrix {
        let mut res: Matrix = Matrix::new(self.height, m.width);
        if self.width == m.height {
            for i in 0..res.width {
                for j in 0..m.height {
                    let mut tmp: f64 = 0.0;
                    for a in 0..self.width {
                        tmp = tmp + self.data[j][a] * m.data[a][i];
                    }
                    res.data[j][i] = tmp;
                }
            }
        } else {
            panic!("Error while doing a dot product: Dimension incompatibility")
        }
        res
    }

    // adds a matrix of Y height and 1 width to a matrix of Y height and X width
    pub fn add_value_to_all_columns(&self, m: &Matrix) -> Matrix {
        assert_eq!(m.width, 1, "The input matrix should have a width of 1");
        assert_eq!(
            m.height, self.height,
            "The 2 matrices should have the same height"
        );
        let mut res: Matrix = Matrix::new(self.height, self.width);

        for r in 0..self.height {
            for c in 0..self.width {
                res.data[r][c] = self.data[r][c] + m.data[r][0];
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
            print!("\n");
        }
        print!("-------------");
        print!("\n");
    }
}
