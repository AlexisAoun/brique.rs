use crate::{
    checkpoint::Checkpoint, layers::Layer, matrix::Matrix, model::Model, optimizer::Optimizer,
};

const DEFAULT_LAMBDA: f64 = 0.001;
const DEFAULT_OPTIMIZER: Optimizer = Optimizer::SGD {
    learning_step: 0.01,
};
const DEFAULT_PRINT_FREQUENCY: usize = 100;
const DEFAULT_SILENT_MODE: bool = false;
const DEFAULT_DEBUG: bool = false;

#[derive(Clone)]
pub struct ModelBuilder {
    layers: Vec<Layer>,
    user_defined_lambda: Option<f64>,
    user_defined_optimizer: Option<Optimizer>,
    checkpoint: Option<Checkpoint>,
    user_defined_print_frequency: Option<usize>,
    user_defined_debug: Option<bool>,
    user_defined_silent_mode: Option<bool>,
}

impl ModelBuilder {
    pub fn new() -> ModelBuilder {
        ModelBuilder {
            layers: vec![],
            user_defined_debug: None,
            user_defined_silent_mode: None,
            user_defined_print_frequency: None,
            user_defined_optimizer: None,
            user_defined_lambda: None,
            checkpoint: None,
        }
    }

    pub fn add_layer(mut self, layer: Layer) -> ModelBuilder {
        self.layers.push(layer);
        self
    }

    pub fn optimizer(mut self, optimizer: Optimizer) -> ModelBuilder {
        self.user_defined_optimizer = Some(optimizer);
        self
    }

    pub fn l2_reg(mut self, lambda: f64) -> ModelBuilder {
        self.user_defined_lambda = Some(lambda);
        self
    }

    pub fn checkpoint(mut self, checkpoint: Checkpoint) -> ModelBuilder {
        self.checkpoint = Some(checkpoint);
        self
    }

    pub fn verbose(mut self, print_frequency: usize, silent_mode: bool) -> ModelBuilder {
        self.user_defined_print_frequency = Some(print_frequency);
        self.user_defined_silent_mode = Some(silent_mode);
        self
    }

    pub fn debug(mut self, debug: bool) -> ModelBuilder {
        self.user_defined_debug = Some(debug);
        self
    }

    pub fn build(self) -> Model {
        assert_ne!(
            self.layers.len(),
            0,
            "Error : No layers have been added to the model"
        );

        let optimizer: Optimizer = match &self.user_defined_optimizer {
            Some(optimizer) => optimizer.clone(),
            None => DEFAULT_OPTIMIZER,
        };

        let lambda: f64 = match self.user_defined_lambda {
            Some(lambda) => lambda,
            None => DEFAULT_LAMBDA,
        };

        Model::init(self.layers.clone(), optimizer, lambda)
    }

    pub fn build_and_train(
        self,
        data: &Matrix,
        labels: &Matrix,
        batch_size: u32,
        epochs: u32,
        validation_dataset_size: usize,
    ) {
        let print_frequency: usize = match &self.user_defined_print_frequency {
            Some(v) => *v,
            None => DEFAULT_PRINT_FREQUENCY,
        };

        let silent_mode: bool = match self.user_defined_silent_mode {
            Some(v) => v,
            None => DEFAULT_SILENT_MODE,
        };

        let debug: bool = match self.user_defined_debug {
            Some(v) => v,
            None => DEFAULT_DEBUG,
        };

        let checkpoint = self.checkpoint.clone();

        let mut model: Model = self.build();
        model.train(
            data,
            labels,
            batch_size,
            epochs,
            validation_dataset_size,
            checkpoint,
            print_frequency,
            debug,
            silent_mode,
        );
    }
}
