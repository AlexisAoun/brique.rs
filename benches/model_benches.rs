#![allow(unused)]
use brique::layers::*;
use brique::matrix::*;
use brique::model::*;
use brique::optimizer::Optimizer;
use brique::spiral::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("end to end model train", |b| {
        b.iter(|| {
            let (data, labels) = generate_spiral_dataset(1000, 3);

            let layer1 = Layer::init(2, 300, true);
            let layer2 = Layer::init(300, 300, true);
            let layer3 = Layer::init(300, 3, false);

            let layers = vec![layer1, layer2, layer3];

            let sgd = Optimizer::SGD {
                learning_step: 0.001,
            };
            let mut model = Model::init(layers, sgd, 0.01);

            model.train(&data, &labels, 50, 2, 200, false, true);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
