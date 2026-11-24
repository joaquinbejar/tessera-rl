// SPDX-License-Identifier: MIT OR Apache-2.0

//! Matching cost on a full book (`BOOK_CAP` resting orders, 16 owners):
//! a limit order that rests without crossing, a market order sweeping
//! every level of one side, a crossing limit order, the best levels and a
//! cancel. Local only, no claim (SPEC 16.1); the per-level index of design
//! section 1 is decided against these numbers.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_markets::limits::{BOOK_CAP, MAX_OPEN};
use tessera_rl_markets::{Book, Fills, Level, Side};

/// A full book: owners 0 to 15, each with 4 bids below 1000 and 4 asks
/// above it, on 32 distinct prices per side.
fn full_book() -> Book {
    let mut book = Book::new();
    let mut fills = Fills::new();
    for owner in 0..16u8 {
        for k in 0..4u32 {
            let offset = u32::from(owner) % 8 * 4 + k + 1;
            let _ = book.limit(owner, Side::Buy, 1000 - offset, 3, &mut fills);
            let _ = book.limit(owner, Side::Sell, 1000 + offset, 3, &mut fills);
        }
    }
    book
}

fn bench(c: &mut Criterion) {
    let full = full_book();
    assert_eq!(full.len(), BOOK_CAP);
    let mut fills = Fills::new();
    // One owner below MAX_OPEN, so a new limit order is accepted.
    let mut spare = full;
    let freed = spare.cancel_all(15);
    assert_eq!(freed, MAX_OPEN);

    c.bench_function("lob_limit_rest_full_book", |b| {
        b.iter(|| {
            let mut book = spare;
            black_box(book.limit(15, Side::Buy, black_box(900), 1, &mut fills))
        });
    });
    c.bench_function("lob_market_sweep_side", |b| {
        b.iter(|| {
            let mut book = spare;
            black_box(book.market(15, Side::Buy, black_box(1024), &mut fills))
        });
    });
    c.bench_function("lob_limit_cross_three_levels", |b| {
        b.iter(|| {
            let mut book = spare;
            black_box(book.limit(15, Side::Buy, black_box(1003), 20, &mut fills))
        });
    });
    c.bench_function("lob_levels_top10_both_sides", |b| {
        let mut out = [Level::default(); 10];
        b.iter(|| {
            let n = full.levels(Side::Buy, &mut out) + full.levels(Side::Sell, &mut out);
            black_box(n)
        });
    });
    c.bench_function("lob_sweep_cost_top10", |b| {
        b.iter(|| black_box(full.sweep(Side::Buy, 15, black_box(100), 10)));
    });
    c.bench_function("lob_cancel_full_book", |b| {
        let seq = full.orders().map(|o| o.seq).max().unwrap_or(0);
        b.iter(|| {
            let mut book = full;
            black_box(book.cancel(15, black_box(seq)))
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
