# T1, sheet A: grid walk

Write this environment as a new crate on the `Environment` trait, starting
from `cargo tessera new <name> --kind env --tessera-path <checkout>` (one
agent, simultaneous turns, the defaults). The task ends when
`cargo tessera check` passes and the facilitator's grader passes.

The grader replays your environment from fixed seeds with a fixed action
script and compares every step with a reference, so follow the rules and the
encoding below exactly.

## Rules

- A 5 x 5 grid. Cells are (row, column), row 0 at the top. Walls stand at
  (1, 1), (1, 3), (3, 1) and (3, 3). The goal is (4, 4).
- One agent. Actions, `Discrete(5)`: 0 stay, 1 up (row - 1), 2 right
  (column + 1), 3 down (row + 1), 4 left (column - 1). A move off the grid or
  into a wall leaves the agent where it is. Every action is legal: the mask
  allows all five, and the no-op is 0.
- Reset: draw exactly one `rng.next_u32()` from the reset's `StepRng`, and
  start on `FREE[draw % 20]`, where `FREE` lists, in row-major order, the 20
  cells that are neither a wall nor the goal.
- Each step: move, then
  - if the agent is on the goal: reward 1.0 and the episode terminates;
  - otherwise reward 0.0, and the episode truncates at step 20, counting
    the steps since reset (configuration field `episode`, default 20).

## Encoding

- Observation: exactly one leaf, `Box(i32, [4])` with bounds 0 to 4: agent
  row, agent column, goal row, goal column.
- The configuration's defaults are the values above.

## Grading interface

End `src/lib.rs` with a type alias naming your environment:

```rust
/// The environment the study's grader runs.
pub type StudyEnv = YourEnvironment;
```

The grader runs `cargo tessera check`, `cargo clippy --all-targets -- -D
warnings`, the conformance replay and a lint check that `src/` has no unsafe
code and no `unwrap`, `expect`, `panic` or unchecked indexing.
