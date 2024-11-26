#![allow(unused)]
use brique::matrix::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let rand_matrix1_10x10: Matrix = Matrix::init_rand(10, 10);
    let rand_matrix2_10x10: Matrix = Matrix::init_rand(10, 10);
    let rand_matrix1_10x1: Matrix = Matrix::init_rand(10, 1);

    let rand_matrix1_100x100: Matrix = Matrix::init_rand(100, 100);
    let rand_matrix2_100x100: Matrix = Matrix::init_rand(100, 100);
    let rand_matrix1_100x1: Matrix = Matrix::init_rand(100, 1);

    let rand_matrix1_1000x1000: Matrix = Matrix::init_rand(1000, 1000);
    let rand_matrix2_1000x1000: Matrix = Matrix::init_rand(1000, 1000);
    let rand_matrix1_1000x1: Matrix = Matrix::init_rand(1000, 1);

    //init rand matrix
    c.bench_function("init rand 10x10", |b| {
        b.iter(|| Matrix::init_rand(black_box(10), black_box(10)))});

    c.bench_function("init rand 100x100", |b| {
        b.iter(|| Matrix::init_rand(black_box(100), black_box(100)))});

    c.bench_function("init rand 1000x1000", |b| {
        b.iter(|| Matrix::init_rand(black_box(1000), black_box(1000)))});

    //dot product
    c.bench_function("dot product 10x10", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(10, 10);
            let rand_matrix2: Matrix = Matrix::init_rand(10, 10);
            black_box(&rand_matrix1).dot(black_box(&rand_matrix2))})
    });

    c.bench_function("dot product 100x100", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(100, 100);
            let rand_matrix2: Matrix = Matrix::init_rand(100, 100);
            black_box(&rand_matrix1).dot(black_box(&rand_matrix2))})
    });

    c.bench_function("dot product 1000x1000", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(1000, 1000);
            let rand_matrix2: Matrix = Matrix::init_rand(1000, 1000);
            black_box(&rand_matrix1).dot(black_box(&rand_matrix2))})
    });

    //add 2 matrices
    c.bench_function("add two matrices 10x10", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(10, 10);
            let rand_matrix2: Matrix = Matrix::init_rand(10, 10);
            black_box(&rand_matrix1).add_two_matrices(black_box(&rand_matrix2))})
    });

    c.bench_function("add two matrices 100x100", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(100, 100);
            let rand_matrix2: Matrix = Matrix::init_rand(100, 100);
            black_box(&rand_matrix1).add_two_matrices(black_box(&rand_matrix2))})
    });

    c.bench_function("add two matrices 1000x1000", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(1000, 1000);
            let rand_matrix2: Matrix = Matrix::init_rand(1000, 1000);
            black_box(&rand_matrix1).add_two_matrices(black_box(&rand_matrix2))})
    });

    //add values to all rows 
    c.bench_function("add values to all rows 10x10", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(1, 10);
            let rand_matrix2: Matrix = Matrix::init_rand(1, 10);
            black_box(&rand_matrix1).add_value_to_all_rows(black_box(&rand_matrix2))})
    });

    c.bench_function("add values to all rows 100x100", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(1, 100);
            let rand_matrix2: Matrix = Matrix::init_rand(1, 100);
            black_box(&rand_matrix1).add_value_to_all_rows(black_box(&rand_matrix2))})
    });

    c.bench_function("add values to all rows 1000x1000", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(1, 1000);
            let rand_matrix2: Matrix = Matrix::init_rand(1, 1000);
            black_box(&rand_matrix1).add_value_to_all_rows(black_box(&rand_matrix2))})
    });
   
    // pow 
    c.bench_function("pow 10x10", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(10, 10);
            black_box(&rand_matrix1).pow(black_box(5))})
    });

    c.bench_function("pow 100x100", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(100, 100);
            black_box(&rand_matrix1).pow(black_box(5))})
    });

    c.bench_function("pow 1000x1000", |b| {
        b.iter(|| {
            let rand_matrix1: Matrix = Matrix::init_rand(1000, 1000);
            black_box(&rand_matrix1).pow(black_box(5))})
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
