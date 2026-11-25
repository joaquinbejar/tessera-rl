# T1, sheet C: simultaneous pickup

Write this environment as a new crate on the `Environment` trait, starting
from `cargo tessera new <name> --kind env --agents 2 --tessera-path
<checkout>` (simultaneous turns, the default). The task ends when
`cargo tessera check` passes and the facilitator's grader passes.

The grader replays your environment from fixed seeds with a fixed action
script and compares every step's rewards, episode flags and masks with a
reference, so follow the rules below exactly. The observation is yours to
design.

## Rules

- A 5 x 5 grid. Cells are (row, column), row 0 at the top. Agent 0 starts at
  (0, 0) and agent 1 at (4, 4). Both act every step.
- Three items. Reset draws one `rng.next_u32()` per item from the reset's
  `StepRng`, in item order 0, 1, 2. Item k goes on `CELLS[draw % 23]`, where
  `CELLS` lists every cell but the two start cells in row-major order. If
  that cell already holds an item, take the next cell of `CELLS`, wrapping
  from the last to the first, until one is free.
- Actions, `Discrete(5)`: 0 stay, 1 up (row - 1), 2 right (column + 1),
  3 down (row + 1), 4 left (column - 1). A move off the grid is illegal: the
  mask, one row per agent, forbids it. Staying is always legal and is the
  no-op. Both agents may stand on the same cell.
- Each step: agent 0 moves, then agent 1. Then each item not yet taken, in
  item order, goes to agent 0 if it stands on the item's cell, else to
  agent 1 if it does: reward 1.0 to the taker for each item it takes. Other
  rewards are 0.0.
- Taking the last item terminates the episode. Otherwise it truncates at
  step 30, counting the steps since reset (configuration field `episode`,
  default 30).

## Requirements

- Partial observation: each agent sees only its own position and the cells
  next to it, not the whole grid. The encoding is yours; keep every value
  within the bounds you declare.
- Store the items in fixed-capacity fields of the state (arrays), and store
  at least one enum in the state with `#[derive(StateEnum)]`.

## Grading interface

End `src/lib.rs` with a type alias naming your environment:

```rust
/// The environment the study's grader runs.
pub type StudyEnv = YourEnvironment;
```

The grader runs `cargo tessera check`, `cargo clippy --all-targets -- -D
warnings`, the conformance replay (rewards, episode flags and masks) and a
lint check that `src/` has no unsafe code and no `unwrap`, `expect`, `panic`
or unchecked indexing.
