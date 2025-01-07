#![allow(unused)]
use brique::parse_test_csv::parse_test_csv;
use brique::{matrix::*, parse_test_csv};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn get_test_matrices() -> Vec<Matrix> {
    parse_test_csv("benches/bench_data.csv".to_string())
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("dot product 100x100", |b| {
        b.iter(|| {
            let matrices: Vec<Matrix> = get_test_matrices();
            black_box(&matrices[0]).dot(black_box(&matrices[1]))
        })
    });

    c.bench_function("add two matrices 100x100", |b| {
        b.iter(|| {
            let matrices: Vec<Matrix> = get_test_matrices();
            black_box(&matrices[0]).add_two_matrices(black_box(&matrices[1]))
        })
    });

    c.bench_function("add values to all rows 100x100", |b| {
        b.iter(|| {
            let matrices: Vec<Matrix> = get_test_matrices();
            black_box(&matrices[0]).add_1d_matrix_to_all_rows(black_box(&matrices[2]))
        })
    });

    c.bench_function("pow 100x100", |b| {
        b.iter(|| {
            let matrices: Vec<Matrix> = get_test_matrices();
            black_box(&matrices[0]).pow(black_box(5))
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
