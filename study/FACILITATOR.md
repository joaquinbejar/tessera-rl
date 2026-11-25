# Facilitator script

## Before the session

1. Clone the repository at the kit's pinned commit and remove
   `study/grading/` from the participant's copy. Keep a full checkout for
   grading.
2. Pick the T1 sheet by rotation (A, B, C) and the T2 variant, then copy the
   variant over the crate: `cp study/tasks/t2-derive/variants/<v>.rs study/tasks/t2-derive/src/lib.rs`.
3. Check that the participant's clone builds offline:
   `cargo build --locked --offline` in `study/tasks/t2-derive` must fail with
   the variant's diagnostic, and `cargo tessera --version` must print the
   kit's version.
4. Create the session record from `study/records/TEMPLATE.md`.

## During the session

- Read the task's start condition aloud, start the timer, and log the time.
- Answer questions only through the help ladder below, and log every use.
- Pause the timer for breaks and log them.
- Stop at the stop condition or the hard cap (study/README.md).

### Help ladder

1. A pointer to a page or a section of the tutorial.
2. A hint that names the area, without the answer ("look at what `reset`
   reads").
3. The answer. The task is then recorded as finished with help.

## T1: grading

When the participant says their environment works and `cargo tessera check`
passes, run the grader on their crate from the full checkout:

```sh
study/grading/grade.sh <their-crate> <a|b|c>
```

T1 stops when every line reads PASS. Paste the grader's output into the
record verbatim. If a line reads FAIL, tell the participant which check
failed, but not why, and keep the timer running.

## T2 and T3: accepting the answer

- **T2:** the cause is the variant's rule (a `bool` field where ADR-0029
  requires `Bool8`; trailing padding without `Reserved<N>`; an enum without
  `#[derive(StateEnum)]`), and the fix compiles.
- **T3:** the faulty item is the `static RESETS` counter and the line in
  `reset` that reads it; the reason is process-global state outside the
  `StepRng`, which carries from one pool to the next. Naming the reported
  comparison alone (`Inline vs Inline`) is not enough.

## After the session

1. The participant fills in the questionnaire (records/TEMPLATE.md).
2. Record the pages they consulted (self-reported), their prior reading of
   the repository, and whether they opened `study/grading/`.
3. Remove identifying details from free text before committing the record.
4. Restore the kit: `git checkout -- study/tasks/t2-derive/src/lib.rs`.
