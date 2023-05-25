// To acces Matrix data : Matrix.data[row][column]

pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub width: u32,
    pub height: u32,
}

impl Matrix {
    pub fn new(height: u32, width: u32) -> Matrix {
        Matrix {
            data: vec![vec![0.0; width.try_into().unwrap()]; height.try_into().unwrap()],
            width,
            height,
        }
    }

    pub fn dot(&self, m: Matrix) -> Matrix {
        let mut res: Matrix = Matrix::new(self.height, m.width);
        if self.width == m.height {
            for i in 0usize..res.width.try_into().unwrap() {
                for j in 0usize..m.height.try_into().unwrap() {
                    let mut tmp: f64 = 0.0;
                    for a in 0usize..self.width.try_into().unwrap() {
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

    pub fn display(&self) {
        print!("\n");
        print!("-------------");
        print!("\n");
        for i in 0..self.height.try_into().unwrap() {
            for j in 0..self.width.try_into().unwrap() {
                print!(" {} |", self.data[i][j]);
            }
            print!("\n");
        }
        print!("-------------");
        print!("\n");
    }
}
