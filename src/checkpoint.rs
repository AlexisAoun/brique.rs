#[derive(Clone, Debug)]
pub enum Checkpoint {
    ValLoss { save_path: String },
    ValAcc { save_path: String },
}
