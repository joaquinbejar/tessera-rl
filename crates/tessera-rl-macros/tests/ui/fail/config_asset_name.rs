// SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;

use tessera_rl_core::{Asset, AssetData, AssetError};
use tessera_rl_macros::EnvConfig;

struct Map;

impl AssetData for Map {
    const MAX_DECODED_BYTES: usize = 0;

    fn decode(_: &[u8]) -> Result<Self, AssetError> {
        Ok(Self)
    }

    fn canonical_bytes(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&[])
    }

    fn bundled(_: &str) -> Option<&'static [u8]> {
        None
    }
}

#[derive(EnvConfig)]
struct C {
    #[config(asset = "Bad Name")]
    map: Asset<Map>,
}

fn main() {}
