use plotters::prelude::*;
use crate::spiral::*;

pub fn draw_test() {
    let (data, labels) = generate_spiral_dataset(100, 3);

    let root_area = BitMapBackend::new("spiral_dataset_vis_2.png", (600, 400))
    .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Scatter Demo", ("sans-serif", 40))
        .build_cartesian_2d(-1000..1000, -1000..1000)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let iter_1 = data.data[0..100].iter().map(|point| ((point[0] * 1000.0) as i32, (point[1] * 1000.0) as i32));
    let iter_2 = data.data[100..200].iter().map(|point| ((point[0] * 1000.0) as i32, (point[1] * 1000.0) as i32));
    let iter_3 = data.data[200..300].iter().map(|point| ((point[0] * 1000.0) as i32, (point[1] * 1000.0) as i32));

    ctx.draw_series(
        iter_1.map(|point| Circle::new(point, 5, &BLUE)),
    )
    .unwrap();

    ctx.draw_series(
        iter_2.map(|point| Circle::new(point, 5, &RED)),
    )
    .unwrap();

    ctx.draw_series(
        iter_3.map(|point| Circle::new(point, 5, &GREEN)),
    )
    .unwrap();
}


