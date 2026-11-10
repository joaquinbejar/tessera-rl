// SPDX-License-Identifier: MIT OR Apache-2.0

//! The grid: a square of `size` cells, row-major, with its mirror, its
//! neighbours and its distance.
//!
//! The 180-degree rotation that maps one team's half onto the other's is
//! index reversal: cell `i` maps to `size * size - 1 - i`. Each team reads
//! the four neighbours of a cell in a fixed order of its own frame, so that
//! the rotation also maps team 0's choices onto team 1's.

#![deny(clippy::float_arithmetic)]

/// Smallest side.
pub const MIN_SIZE: u16 = 8;
/// Largest side.
pub const MAX_SIZE: u16 = 32;
/// Cells of the largest grid; fixed arrays use it.
pub const MAX_CELLS: usize = (MAX_SIZE as usize) * (MAX_SIZE as usize);

/// What a cell is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Terrain {
    /// Passable and buildable.
    Open = 0,
    /// Impassable.
    Wall = 1,
    /// An ore node: impassable, harvested from a 4-adjacent cell.
    Ore = 2,
}

impl Terrain {
    /// The terrain of a stored byte; `None` for any other value.
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Open),
            1 => Some(Self::Wall),
            2 => Some(Self::Ore),
            _ => None,
        }
    }
}

/// One of the two teams; team 0 is agent slot 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Team {
    /// Agent slot 0.
    Zero,
    /// Agent slot 1.
    One,
}

impl Team {
    /// The other team.
    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::Zero => Self::One,
            Self::One => Self::Zero,
        }
    }
}

/// A step to a 4-neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    /// Row - 1.
    North,
    /// Column + 1.
    East,
    /// Row + 1.
    South,
    /// Column - 1.
    West,
}

impl Dir {
    /// The fixed neighbour order in `team`'s frame: team 1's is the
    /// 180-degree image of team 0's.
    #[must_use]
    pub const fn order(team: Team) -> [Self; 4] {
        match team {
            Team::Zero => [Self::North, Self::East, Self::South, Self::West],
            Team::One => [Self::South, Self::West, Self::North, Self::East],
        }
    }

    /// The direction after the 180-degree rotation.
    #[must_use]
    pub const fn mirrored(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::East => Self::West,
            Self::South => Self::North,
            Self::West => Self::East,
        }
    }
}

/// A validated side and the cell arithmetic on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    size: u16,
}

impl Grid {
    /// A grid of side `size`; `None` outside `MIN_SIZE..=MAX_SIZE`.
    #[must_use]
    pub const fn new(size: u16) -> Option<Self> {
        if size >= MIN_SIZE && size <= MAX_SIZE {
            Some(Self { size })
        } else {
            None
        }
    }

    /// The side.
    #[must_use]
    pub const fn size(self) -> u16 {
        self.size
    }

    /// The number of cells, at most [`MAX_CELLS`].
    #[must_use]
    pub const fn cells(self) -> u16 {
        self.size * self.size
    }

    /// The cell at column `x` and row `y`, if inside.
    #[must_use]
    pub const fn index(self, x: u16, y: u16) -> Option<u16> {
        if x < self.size && y < self.size {
            Some(y * self.size + x)
        } else {
            None
        }
    }

    /// Column and row of `cell`.
    #[must_use]
    pub const fn coords(self, cell: u16) -> (u16, u16) {
        (cell % self.size, cell / self.size)
    }

    /// The cell after the 180-degree rotation (rule 9 of the design).
    #[must_use]
    pub const fn mirror(self, cell: u16) -> u16 {
        self.cells() - 1 - cell
    }

    /// The 4-neighbour of `cell` in `dir`, if inside.
    #[must_use]
    pub const fn step(self, cell: u16, dir: Dir) -> Option<u16> {
        let (x, y) = self.coords(cell);
        match dir {
            Dir::North if y > 0 => Some(cell - self.size),
            Dir::South if y + 1 < self.size => Some(cell + self.size),
            Dir::West if x > 0 => Some(cell - 1),
            Dir::East if x + 1 < self.size => Some(cell + 1),
            _ => None,
        }
    }

    /// The 4-neighbours of `cell` in `team`'s fixed order.
    #[must_use]
    pub fn neighbours(self, cell: u16, team: Team) -> [Option<u16>; 4] {
        Dir::order(team).map(|d| self.step(cell, d))
    }

    /// Chebyshev distance: the one metric of the environment, for sight
    /// and for range (rule 11), so range 1 includes the diagonals.
    #[must_use]
    pub const fn chebyshev(self, a: u16, b: u16) -> u16 {
        let (ax, ay) = self.coords(a);
        let (bx, by) = self.coords(b);
        let dx = ax.abs_diff(bx);
        let dy = ay.abs_diff(by);
        if dx > dy { dx } else { dy }
    }

    /// Whether `a` and `b` share an edge.
    #[must_use]
    pub const fn adjacent(self, a: u16, b: u16) -> bool {
        let (ax, ay) = self.coords(a);
        let (bx, by) = self.coords(b);
        ax.abs_diff(bx) + ay.abs_diff(by) == 1
    }
}

/// The index of a target window's cell after the 180-degree rotation: in
/// the row-major window of odd side `w`, index `i` maps to `w * w - 1 - i`,
/// which is offset `(dx, dy)` mapping to `(-dx, -dy)` (rule 9).
#[must_use]
pub const fn mirror_window(i: u16, w: u16) -> u16 {
    w * w - 1 - i
}
