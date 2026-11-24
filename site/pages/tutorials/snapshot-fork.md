# Snapshot, restore and fork

Environment state is plain data, so copying it is cheap. That makes search on
the real environment practical: copy a state, try different actions on the
copies, and compare the outcomes.

- `env.snapshot_all()` and `env.restore_pool(snapshot)` save and restore the
  whole pool, exactly.
- `env.fork(src, dst)` copies instance `src` into the instances `dst`.
- `env.state_hash(i)` is a hash of instance `i`'s state, handy to check that
  two states are equal.

<!-- snippet: snapshot_fork.py -->

Restoring is exact: stepping again from a restored state with the same
actions gives the same outputs, bit for bit, on every platform.
