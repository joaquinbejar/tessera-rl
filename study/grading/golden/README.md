# Reference trajectories

`<sheet>.txt` is the trajectory of the reference for T1 sheet `<sheet>`,
recorded by `STUDY_BLESS=1 study/grading/grade.sh study/grading/reference/<crate> <sheet>`
and compared by the conformance check (`study/grading/conform/conform.rs`).
Re-record one only when its sheet changes, and say so in the kit's change
record.
