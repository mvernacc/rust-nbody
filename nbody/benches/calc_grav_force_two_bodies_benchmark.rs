use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nbody::gravity;

pub fn criterion_benchmark_calc_grav_force_two_bodies(c: &mut Criterion) {
    c.bench_function("calc_grav_force_two_bodies", |b| {
        b.iter(|| {
            gravity::calc_grav_force_two_bodies(
                black_box(1e20),
                black_box(2e20),
                black_box([1e9, 0.5e9, 0.6e9]),
                black_box([1e3, 2e3, 1.2e3]),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark_calc_grav_force_two_bodies);
criterion_main!(benches);
