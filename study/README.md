# The authoring study (SPEC 10.3)

This directory holds the protocol, the tasks and the grading kit of the
authoring study that SPEC 10.3 requires before 1.0 (#639; design:
docs/reviews/639/DESIGN-r2.md, approved). SPEC 10.3 reads:

> Before 1.0, at least three developers who did not design the engine write
> an environment from the tutorial. We record time to first working
> environment, time to understand a seeded derive error, and time to locate a
> seeded determinism divergence with the provided tools, and publish the
> results.

Recruiting the participants and publishing the results are the owner's
decisions and are not part of this kit.

## What is measured

The three SPEC timings are the primary measures; there is no pass threshold.
As a secondary reading, the study reports against the project's informal
target, "a new environment in an afternoon, without segfaults": T1 in at most
4 hours, and no crash outside Rust's safe error paths.

| Task | Start | Stop | Hard cap |
| --- | --- | --- | --- |
| T1: first working environment | The participant opens the task sheet. | `cargo tessera check` passes on their crate and the facilitator confirms that `grading/grade.sh` passes for the sheet. | 6 h |
| T2: understand a seeded derive error | The participant runs `cargo build` in `tasks/t2-derive`. | They state the cause in one sentence, the facilitator records it as correct, and their fix compiles. | 45 min |
| T3: locate a seeded determinism divergence | The participant runs `cargo tessera check` in `tasks/t3-divergence` and sees the failure. | They name the faulty item (the `static` and the line in `reset` that reads it) and the reason (process-global state outside `StepRng`), and the facilitator records it as correct. | 90 min |

Timers run in wall minutes. A break pauses the timer and is logged. At a hard
cap the task stops and is recorded as not finished, with the time.

## The tasks

- **T1** is one of three rule sheets, assigned in rotation A, B, C so that
  three participants cover all three:
  - [`tasks/t1-a-gridwalk.md`](tasks/t1-a-gridwalk.md): a single-agent grid walk;
  - [`tasks/t1-b-nim.md`](tasks/t1-b-nim.md): two-player sequential Nim with an action mask;
  - [`tasks/t1-c-pickup.md`](tasks/t1-c-pickup.md): simultaneous two-agent pickup with partial observation.
- **T2** is `tasks/t2-derive/`, the scaffold with one seeded derive error. The
  facilitator picks one variant from `tasks/t2-derive/variants/` per
  participant and copies it over `src/lib.rs` before the session.
- **T3** is `tasks/t3-divergence/`, the scaffold with one seeded
  determinism divergence. The pre-declared observation is the comparison kind
  the check reports first, `Inline vs Inline (same process)`, not a call
  number: the seed's call and instance numbers vary from run to run.

## Allowed material

- The tutorial
  [`site/pages/tutorials/writing-an-environment-trait.md`](../site/pages/tutorials/writing-an-environment-trait.md)
  and the pages it links to;
- `cargo tessera` (`new --kind env`, `check`, `bench`), the compiler, clippy
  and the participant's own editor;
- the Rust API documentation (`cargo doc`) of the Tessera crates.

The participant works in a clone of this repository at the kit's pinned
commit, with `study/grading/` removed from their copy. "Hidden" means not
handed out: the repository is public, so the record notes whether the
participant opened `study/grading/` anyway.

## Eligibility (owner to confirm)

Default, pending the owner's confirmation because it concerns recruiting:
participants who have read the repository may take part if they did not
design the engine. Their prior reading is recorded.

## Versions and pooling

Every record names the `cargo-tessera` version and the repository commit
that `grading/grade.sh` prints. The kit requires a commit at or after #652,
which numbers every check report by absolute call (#645). Results from
different `cargo-tessera` versions or from both sides of a change to the
scaffold's check report are never pooled.

## Records

One record per session, `records/<session-id>.md`, from
[`records/TEMPLATE.md`](records/TEMPLATE.md). The id is an opaque code such
as `P1`, never a name. No names, emails, employers, recordings or machine
details are kept in the repository. Consent and the key from session id to
person are kept by the owner, outside the repository.

A dry run by one internal teammate, labelled `DRY-1`, checks the kit and is
not a study result.

## The grading kit

[`grading/`](grading/) holds the reference implementations, their recorded
trajectories, the grader and the kit's acceptance run:

- `grading/grade.sh <crate> <a|b|c|none>`: the offline grader. It prints one
  PASS or FAIL line per check (`check`, `clippy`, `conformance`, `safety`) and
  nothing else, so its output replays byte for byte.
- `grading/acceptance.sh`: the kit's own acceptance (design r2 section 8), on
  Linux.

Every crate here has its own workspace and a `Cargo.lock` derived from the
repository's, and depends on the Tessera crates of this checkout by path; it
is not part of the repository's workspace. The facilitator script is
[`FACILITATOR.md`](FACILITATOR.md).
