struct Matrice {
    data: Vec<Vec<f64>>,
    width: u32,
    height: u32
}

impl Matrice {
   fn new(width: u32, height: u32)->Matrice{
       Matrice {
        data: vec![vec![0.0; height.try_into().unwrap()]; width.try_into().unwrap()],
        width,
        height
       }
   }

   fn dot(&self, m:Matrice)->Matrice {
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

   fn display(&self) {
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

fn main() {
    let mut m1: Matrice = Matrice::new(2,2);
    m1.data[0][0] = -3242.213;
    m1.data[1][0] = 1242356.4245;
    m1.data[0][1] = 41466.9088;
    m1.data[1][1] = 0.0;
    let mut m2: Matrice = Matrice::new(2,3);
    m2.data[0][0] = 898.0;
    m2.data[1][0] = -222.3;
    m2.data[0][1] = -2467.9098;
    m2.data[1][1] = 1356770.0;
    m2.data[0][2] = -696969.69696;
    m2.data[1][2] = 10.0;
    m1.display();
    m2.display();

    let m3 = m2.dot(m1);
    m3.display();

}
