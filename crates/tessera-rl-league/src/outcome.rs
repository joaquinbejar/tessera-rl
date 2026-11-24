// SPDX-License-Identifier: MIT OR Apache-2.0

//! The outcome of one episode for the learner slot (ADR-0088 item 4.1).

/// Win, draw or loss, from the learner's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Outcome {
    /// The learner lost.
    Loss = 0,
    /// A draw.
    Draw = 1,
    /// The learner won.
    Win = 2,
}

impl Outcome {
    /// The default rule: the sign of the learner's episode return minus the
    /// opponents' mean return. Equal returns, or any NaN, are a draw.
    #[must_use]
    pub fn from_returns(learner: f64, opponents_mean: f64) -> Self {
        let d = learner - opponents_mean;
        if d > 0.0 {
            Self::Win
        } else if d < 0.0 {
            Self::Loss
        } else {
            Self::Draw
        }
    }

    /// The learner's score: 1, 0.5 or 0.
    #[must_use]
    pub const fn score(self) -> f64 {
        match self {
            Self::Win => 1.0,
            Self::Draw => 0.5,
            Self::Loss => 0.0,
        }
    }

    /// The outcome of a stored byte; `None` for an unknown one.
    #[must_use]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Loss),
            1 => Some(Self::Draw),
            2 => Some(Self::Win),
            _ => None,
        }
    }
}
