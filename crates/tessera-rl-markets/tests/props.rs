// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property tests of the book (`docs/reviews/381/DESIGN.md`, section 7):
//! after any sequence of limit, market and cancel operations, lots are
//! conserved, the book is never crossed, cash plus inventory valued at the
//! fill price is conserved by every fill, no agent holds more than
//! `MAX_OPEN` resting orders, and the book never overflows.

use proptest::prelude::*;
use tessera_rl_markets::limits::{BOOK_CAP, MAX_AGENTS, MAX_OPEN};
use tessera_rl_markets::{Account, Book, BookError, Fills, Order, Side, apply_fill};

const CASH: i64 = 1 << 40;

#[derive(Debug, Clone, Copy)]
enum Op {
    Limit {
        owner: u8,
        side: Side,
        price: u32,
        qty: u32,
    },
    Market {
        owner: u8,
        side: Side,
        qty: u32,
    },
    Cancel {
        owner: u8,
        index: usize,
    },
}

fn side() -> impl Strategy<Value = Side> {
    prop_oneof![Just(Side::Buy), Just(Side::Sell)]
}

/// Prices in a narrow band so that orders cross often; up to 16 owners.
fn op() -> impl Strategy<Value = Op> {
    let owner = 0..16u8;
    prop_oneof![
        5 => (owner.clone(), side(), 90..=110u32, 1..=20u32)
            .prop_map(|(owner, side, price, qty)| Op::Limit { owner, side, price, qty }),
        2 => (owner.clone(), side(), 1..=40u32)
            .prop_map(|(owner, side, qty)| Op::Market { owner, side, qty }),
        2 => (owner, 0..MAX_OPEN).prop_map(|(owner, index)| Op::Cancel { owner, index }),
    ]
}

fn book_lots(book: &Book) -> u64 {
    book.orders().map(|o| u64::from(o.qty)).sum()
}

fn valued(accounts: &[Account], price: u32) -> i128 {
    accounts
        .iter()
        .map(|a| i128::from(a.cash) + i128::from(a.inventory) * i128::from(price))
        .sum()
}

/// Applies `op` and checks the per-operation properties.
fn apply(
    book: &mut Book,
    accounts: &mut [Account; MAX_AGENTS],
    fills: &mut Fills,
    op: Op,
) -> Result<(), TestCaseError> {
    let before = *book;
    let lots_before = book_lots(book);
    let (owner, qty, result) = match op {
        Op::Limit {
            owner,
            side,
            price,
            qty,
        } => (owner, qty, Some(book.limit(owner, side, price, qty, fills))),
        Op::Market { owner, side, qty } => (owner, qty, Some(book.market(owner, side, qty, fills))),
        Op::Cancel { owner, index } => {
            let mut own = [Order::default(); MAX_OPEN];
            let n = book.own_orders(owner, &mut own);
            if index < n {
                let target = own.get(index).map_or(u32::MAX, |o| o.seq);
                let cancelled = book.cancel(owner, target);
                prop_assert_eq!(cancelled.map(|o| o.seq), Some(target));
                let lots = cancelled.map_or(0, |o| u64::from(o.qty));
                prop_assert_eq!(book_lots(book), lots_before - lots);
            } else {
                prop_assert_eq!(*book, before);
            }
            (owner, 0, None)
        }
    };
    let Some(result) = result else {
        return Ok(());
    };
    let out = match result {
        Ok(out) => out,
        Err(e) => {
            prop_assert_eq!(e, BookError::OpenLimit(owner));
            prop_assert_eq!(*book, before);
            return Ok(());
        }
    };
    // The fills: their lots add up, never between an agent and itself, and
    // each conserves total cash plus inventory valued at its price.
    let mut filled = 0u32;
    for fill in fills.as_slice() {
        prop_assert_ne!(fill.maker, fill.taker);
        filled += fill.qty;
        let total = valued(accounts, fill.price);
        prop_assert!(apply_fill(accounts, fill));
        prop_assert_eq!(valued(accounts, fill.price), total);
    }
    prop_assert_eq!(filled, out.filled);
    // Lots conserved: incoming = filled + rested + dropped, and the book
    // lost exactly the filled and self-trade-cancelled lots.
    let rested = out
        .rested
        .and_then(|seq| book.order(seq))
        .map_or(0, |o| o.qty);
    prop_assert_eq!(qty, out.filled + rested + out.dropped);
    prop_assert_eq!(
        book_lots(book) + u64::from(out.filled) + u64::from(out.stp_lots),
        lots_before + u64::from(rested)
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn test_book_any_op_sequence_keeps_invariants(ops in proptest::collection::vec(op(), 1..400)) {
        let mut book = Book::new();
        let mut accounts = [Account::with_cash(CASH); MAX_AGENTS];
        let mut fills = Fills::new();
        for op in ops {
            apply(&mut book, &mut accounts, &mut fills, op)?;
            prop_assert!(!book.is_crossed());
            prop_assert!(book.len() <= BOOK_CAP);
            for owner in 0..16u8 {
                prop_assert!(book.open_count(owner) <= MAX_OPEN);
                let resting = book.exposure(owner).map(|e| e.total_orders());
                prop_assert_eq!(resting, u32::try_from(book.open_count(owner)).ok());
            }
            // Lots and cash only move between agents.
            prop_assert_eq!(accounts.iter().map(|a| i64::from(a.inventory)).sum::<i64>(), 0);
            prop_assert_eq!(
                accounts.iter().map(|a| i128::from(a.cash)).sum::<i128>(),
                i128::from(CASH) * 16
            );
        }
    }

    #[test]
    fn test_book_sweep_matches_market_fill_cost(
        ops in proptest::collection::vec(op(), 0..200),
        owner in 0..16u8,
        taker in side(),
        qty in 1..=60u32,
    ) {
        let mut book = Book::new();
        let mut accounts = [Account::with_cash(CASH); MAX_AGENTS];
        let mut fills = Fills::new();
        for op in ops {
            apply(&mut book, &mut accounts, &mut fills, op)?;
        }
        let sweep = book.sweep(taker, owner, qty, BOOK_CAP);
        let out = book.market(owner, taker, qty, &mut fills);
        prop_assert_eq!(out.map(|o| o.filled), Ok(sweep.lots));
        let cost: u64 = fills
            .as_slice()
            .iter()
            .map(|f| u64::from(f.qty) * u64::from(f.price))
            .sum();
        prop_assert_eq!(cost, sweep.cost);
        let last = fills.as_slice().last().map(|f| f.price);
        prop_assert_eq!(last, sweep.worst);
    }
}
