// SPDX-License-Identifier: MIT OR Apache-2.0

//! The canonical decoder over core types: arbitrary bytes decode to a
//! typed error or to a value that re-encodes to exactly the same bytes.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tessera_rl_core::encoding::{Canonical, decode_from_slice, encode_to_vec};
use tessera_rl_core::{Bool8, NoAdaptive, RecordHeader};
use tessera_rl_classic::{CartPoleState, TagState};
use tessera_rl_num::{Fx32, Fx64};

fn round_trip<T: Canonical>(data: &[u8]) {
    if let Ok(v) = decode_from_slice::<T>(data) {
        assert_eq!(encode_to_vec(&v), data);
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&which, rest)) = data.split_first() else {
        return;
    };
    match which % 7 {
        0 => round_trip::<Bool8>(rest),
        1 => round_trip::<Fx32>(rest),
        2 => round_trip::<Fx64>(rest),
        3 => round_trip::<RecordHeader<1, NoAdaptive>>(rest),
        4 => round_trip::<RecordHeader<8, NoAdaptive>>(rest),
        5 => round_trip::<CartPoleState>(rest),
        _ => round_trip::<TagState>(rest),
    }
});
