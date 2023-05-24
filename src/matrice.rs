pub struct Matrice {
    pub data: Vec<Vec<f64>>,
    pub width: u32,
    pub height: u32
}

impl Matrice {
   pub fn new(width: u32, height: u32)->Matrice{
       Matrice {
        data: vec![vec![0.0; height.try_into().unwrap()]; width.try_into().unwrap()],
        width,
        height
       }
   }

   pub fn dot(&self, m:Matrice)->Matrice {
       let mut res: Matrice = Matrice::new(m.width, self.height);
       if self.width == m.height {
            for i in 0usize..res.width.try_into().unwrap() {
                for j in 0usize..m.height.try_into().unwrap() {
                    let mut tmp: f64 = 0.0;
                    for a in 0usize..self.width.try_into().unwrap() {
                        tmp = tmp + self.data[a][j]*m.data[i][a];
                    }
                    res.data[i][j] = tmp;
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
                print!(" {} |",self.data[j][i]);
            }
            print!("\n");
        }
        print!("-------------");
        print!("\n");
   }
}
