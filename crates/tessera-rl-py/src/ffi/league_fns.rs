// SPDX-License-Identifier: MIT OR Apache-2.0

//! The module-level functions of `tessera_rl.League` (ADR-0088 section 10,
//! #384 head E3): keyword arguments only.

use tessera_rl_pyffi::ffi::capi::{CResult, Gil, Mine, Ref};
use tessera_rl_pyffi::ffi::module_def::{FunctionKw, FunctionRow};

macro_rules! league_fn {
    ($ty:ident, $f:ident, $name:literal) => {
        struct $ty;
        impl FunctionKw for $ty {
            fn call<'py>(
                g: Gil<'py>,
                args: Ref<'py>,
                kwargs: Option<Ref<'py>>,
            ) -> CResult<Mine<'py>> {
                crate::api::no_positional(g, args, $name)?;
                crate::league_api::$f(g, kwargs)
            }
        }
    };
}

league_fn!(LeagueNew, league_new, "league_new");
league_fn!(LeagueLoad, league_load, "league_load");
league_fn!(LeagueSave, league_save, "league_save");
league_fn!(LeagueExport, league_export, "league_export");
league_fn!(LeagueDrop, league_drop, "league_drop");

/// The rows, for the module's function table.
pub(super) const ROWS: [FunctionRow; 5] = [
    FunctionRow::kw::<LeagueNew>(
        c"league_new",
        c"league_new($module, *, spec)\n--\n\nA shared league from its configuration string; returns its handle.",
    ),
    FunctionRow::kw::<LeagueLoad>(
        c"league_load",
        c"league_load($module, *, data)\n--\n\nA shared league from SharedLeague::save bytes; returns its handle.",
    ),
    FunctionRow::kw::<LeagueSave>(
        c"league_save",
        c"league_save($module, *, league)\n--\n\nThe league's canonical bytes, between rounds.",
    ),
    FunctionRow::kw::<LeagueExport>(
        c"league_export",
        c"league_export($module, *, league)\n--\n\nThe league's JSON export.",
    ),
    FunctionRow::kw::<LeagueDrop>(
        c"league_drop",
        c"league_drop($module, *, league)\n--\n\nForgets a league handle.",
    ),
];
