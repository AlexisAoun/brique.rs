#[derive(Clone)]
pub enum Optimizer {
    SGD {
        learning_step: f64,
    },
    Adam {
        learning_step: f64,
        beta1: f64,
        beta2: f64,
    },
}
