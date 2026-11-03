# Container side: the measurement, on the registered checkpoint sha only.
set -eu
trap 'chown -R "$HOST_UID:$HOST_GID" /work/tessera /out' EXIT
R="uv run --locked --python 3.11 --group test --group torch"
test -n "$CKPT_SHA"
got=$($R python -c 'import sys; sys.path.insert(0, "examples/alphazero_connect4"); import ckpt; print(ckpt.manifest("/out/S.pt")["sha256"])')
test "$got" = "$CKPT_SHA"
$R python examples/alphazero_connect4/group_bench.py --checkpoint /out/S.pt --out /out/gb --source-sha "$S1" \
  --ledger /out/ledger.json --cores 32 --budget-core-hours 24 --wall-cap-s 2700 --cell-allowance-s 30
