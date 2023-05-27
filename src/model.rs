use crate::layers::*;

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    lambda: f64
}
