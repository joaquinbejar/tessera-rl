// SPDX-License-Identifier: MIT OR Apache-2.0

//! One agent's inventory and cash, and how a fill moves them
//! (`docs/reviews/381/DESIGN.md`, section 1).
//!
//! Cash is in tick-lots (`i64`), inventory in lots (`i32`). The config
//! bounds (`LobConfig::validate`) exclude overflow on every path, so the
//! arithmetic here is plain, as in the rest of the kernel.

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::Reserved;
use tessera_rl_macros::EnvState;

use crate::fill::Fill;
use crate::limits::MAX_LOT;

/// One agent's position.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Account {
    /// Cash, in tick-lots.
    pub cash: i64,
    /// Inventory, in lots; negative when short.
    pub inventory: i32,
    /// Zero.
    pub pad: Reserved<4>,
}

const _: () = assert!(size_of::<Account>() == 16);

impl Account {
    /// An account holding `cash` and no inventory.
    #[must_use]
    #[inline]
    pub const fn with_cash(cash: i64) -> Self {
        Self {
            cash,
            inventory: 0,
            pad: Reserved::ZERO,
        }
    }

    /// Cash plus inventory valued at `price`, in tick-lots.
    #[must_use]
    #[inline]
    pub fn equity(&self, price: u32) -> i64 {
        self.cash + i64::from(self.inventory) * i64::from(price)
    }
}

/// Applies `fill` to the buyer's and the seller's accounts in `accounts`,
/// indexed by agent slot. Returns `false`, changing nothing, if either
/// owner has no account or the buyer is the seller (self-trade prevention
/// makes both impossible).
#[inline]
pub fn apply_fill(accounts: &mut [Account], fill: &Fill) -> bool {
    let (buyer, seller) = (usize::from(fill.buyer()), usize::from(fill.seller()));
    let valid = buyer != seller && buyer < accounts.len() && seller < accounts.len();
    debug_assert!(valid, "a fill between invalid or equal owners");
    if !valid {
        return false;
    }
    let notional = fill.notional();
    // `fill.qty <= MAX_LOT = 2^10`: lossless. A broken invariant must not
    // move cash with zero lots unnoticed (review of PR 1, nit 1).
    debug_assert!(
        fill.qty <= MAX_LOT,
        "a fill of more than MAX_LOT lots reached the accounts"
    );
    let lots = i32::try_from(fill.qty).unwrap_or(0);
    if let Some(b) = accounts.get_mut(buyer) {
        b.cash -= notional;
        b.inventory += lots;
    }
    if let Some(s) = accounts.get_mut(seller) {
        s.cash += notional;
        s.inventory -= lots;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{Account, apply_fill};
    use crate::fill::Fill;
    use crate::order::Side;

    #[test]
    fn test_apply_fill_moves_cash_and_lots_conserving_totals() {
        let mut acc = [Account::with_cash(1000); 3];
        let fill = Fill {
            price: 7,
            qty: 3,
            maker: 2,
            taker: 0,
            taker_side: Side::Sell,
            ..Fill::default()
        };
        assert!(apply_fill(&mut acc, &fill));
        assert_eq!(acc.get(2).map(|a| (a.cash, a.inventory)), Some((979, 3)));
        assert_eq!(acc.first().map(|a| (a.cash, a.inventory)), Some((1021, -3)));
        assert_eq!(acc.iter().map(|a| a.cash).sum::<i64>(), 3000);
        assert_eq!(acc.iter().map(|a| a.inventory).sum::<i32>(), 0);
        assert_eq!(acc.get(2).map(|a| a.equity(10)), Some(1009));
    }
}
