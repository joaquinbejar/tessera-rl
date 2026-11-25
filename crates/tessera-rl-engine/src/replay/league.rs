// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `League` event body (tag 12; ADR-0088 section 8, erratum 3; #384
//! head D): a whole [`LeagueView`], canonical, all integers little-endian:
//!
//! ```text
//! league := digest:u64 last:u32 count:u32 kind*count threshold:u64*count
//! kind   := 0:u8 tag:u64                ; External
//!         | 1:u8                        ; Heuristic
//!         | 2:u8 content_hash:[u8; 16]  ; Onnx
//! ```
//!
//! `count` is 1 to [`MAX_LEAGUE_POLICIES`], so a body is at most
//! `16 + 25 x count` bytes (about 1.6 MiB at the cap). Equal views have
//! equal bodies, so "the view changed" (ADR-0088 erratum 3) is a byte
//! comparison of bodies. A decoded view is rebuilt with [`LeagueView::new`],
//! so a body that does not make a valid view is a typed error.

use crate::league::{LeagueView, OpponentKind};

use super::ReplayError;
use super::format::Body;

/// Most policies a `League` body holds.
pub const MAX_LEAGUE_POLICIES: u32 = 1 << 16;

const EXTERNAL: u8 = 0;
const HEURISTIC: u8 = 1;
const ONNX: u8 = 2;

/// Appends `view`'s canonical body.
///
/// # Errors
///
/// [`ReplayError::Unsupported`] for a view of more than
/// [`MAX_LEAGUE_POLICIES`] policies.
pub(crate) fn push_view(out: &mut Vec<u8>, view: &LeagueView) -> Result<(), ReplayError> {
    let count = u32::try_from(view.kinds().len())
        .ok()
        .filter(|&n| n <= MAX_LEAGUE_POLICIES)
        .ok_or(ReplayError::Unsupported {
            feature: "a league of more than 65,536 policies in a replay file",
        })?;
    out.extend_from_slice(&view.population_digest().to_le_bytes());
    out.extend_from_slice(&view.last().to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    for kind in view.kinds() {
        match kind {
            OpponentKind::External { tag } => {
                out.push(EXTERNAL);
                out.extend_from_slice(&tag.to_le_bytes());
            }
            OpponentKind::Heuristic => out.push(HEURISTIC),
            OpponentKind::Onnx { content_hash } => {
                out.push(ONNX);
                out.extend_from_slice(content_hash);
            }
        }
    }
    for t in view.thresholds() {
        out.extend_from_slice(&t.to_le_bytes());
    }
    Ok(())
}

/// `view`'s canonical body.
///
/// # Errors
///
/// As [`push_view`].
pub(crate) fn view_bytes(view: &LeagueView) -> Result<Vec<u8>, ReplayError> {
    let mut out = Vec::new();
    push_view(&mut out, view)?;
    Ok(out)
}

/// Decodes a `League` body of event `event_index`.
///
/// # Errors
///
/// [`ReplayError::Malformed`] for a truncated or trailing body, a count
/// out of range, an unknown kind or a view [`LeagueView::new`] refuses.
pub(crate) fn read_view(body: &[u8], event_index: u64) -> Result<LeagueView, ReplayError> {
    let mut b = Body::new(body, event_index);
    let digest = b.u64("league digest")?;
    let last = b.u32("league last")?;
    // At least one byte of kind and eight of threshold per policy.
    let count = b.count(9, "league count")?;
    if count == 0 || count > MAX_LEAGUE_POLICIES as usize {
        return Err(b.malformed("league count"));
    }
    let mut kinds = Vec::new();
    kinds
        .try_reserve_exact(count)
        .map_err(|_| ReplayError::Allocation)?;
    for _ in 0..count {
        let kind = match b.u8("league kind")? {
            EXTERNAL => OpponentKind::External {
                tag: b.u64("league external tag")?,
            },
            HEURISTIC => OpponentKind::Heuristic,
            ONNX => {
                let hash = b.take(16, "league content hash")?;
                let mut content_hash = [0u8; 16];
                content_hash.copy_from_slice(hash);
                OpponentKind::Onnx { content_hash }
            }
            _ => return Err(b.malformed("league kind")),
        };
        kinds.push(kind);
    }
    let mut thresholds = Vec::new();
    thresholds
        .try_reserve_exact(count)
        .map_err(|_| ReplayError::Allocation)?;
    for _ in 0..count {
        thresholds.push(b.u64("league threshold")?);
    }
    b.finish()?;
    LeagueView::new(kinds, thresholds, Some(last), digest).map_err(|_| ReplayError::Malformed {
        event_index,
        field: "league view",
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{read_view, view_bytes};
    use crate::league::{LeagueView, OpponentKind};

    fn view() -> LeagueView {
        LeagueView::new(
            vec![
                OpponentKind::External { tag: 7 },
                OpponentKind::Heuristic,
                OpponentKind::Onnx {
                    content_hash: [3; 16],
                },
            ],
            vec![1 << 62, 1 << 63, u64::MAX],
            Some(2),
            0xfeed,
        )
        .unwrap()
    }

    #[test]
    fn test_league_body_round_trips_and_is_canonical() {
        let v = view();
        let bytes = view_bytes(&v).unwrap();
        assert_eq!(bytes.len(), 16 + (1 + 8) + 1 + (1 + 16) + 3 * 8);
        assert_eq!(read_view(&bytes, 4).ok(), Some(v.clone()));
        assert_eq!(view_bytes(&v).unwrap(), bytes);
    }

    #[test]
    fn test_league_body_refuses_bad_bodies() {
        let good = view_bytes(&view()).unwrap();
        let bad = |b: &[u8]| read_view(b, 9).is_err();
        assert!(bad(&[]));
        assert!(bad(&good[..good.len() - 1]));
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(bad(&trailing));
        let mut kind = good.clone();
        kind[16] = 9;
        assert!(bad(&kind));
        let mut zero = good.clone();
        zero[12..16].copy_from_slice(&0u32.to_le_bytes());
        assert!(bad(&zero));
        // The last threshold is not the maximum: no drawable entry.
        let mut last = good;
        let n = last.len();
        last[n - 8..].copy_from_slice(&5u64.to_le_bytes());
        assert!(bad(&last));
    }

    proptest! {
        /// Any bytes decode to a view or a typed error, never a panic.
        #[test]
        fn prop_league_body_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
            let _ = read_view(&bytes, 0);
        }
    }
}
